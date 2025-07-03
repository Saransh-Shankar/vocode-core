import asyncio
import base64
import json
import os
from typing import Optional
import io

import soundfile as sf
from google.cloud import aiplatform
from loguru import logger

from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import CambaiVertexSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.utils import convert_wav


class CambaiVertexSynthesizerException(Exception):
    pass


class CambaiVertexSynthesizer(BaseSynthesizer[CambaiVertexSynthesizerConfig]):
    def __init__(self, synthesizer_config: CambaiVertexSynthesizerConfig):
        super().__init__(synthesizer_config)
        
        # Validate required configuration
        if not synthesizer_config.project_id:
            raise ValueError("project_id is required for CambaiVertexSynthesizer")
        if not synthesizer_config.endpoint_id:
            raise ValueError("endpoint_id is required for CambaiVertexSynthesizer")
        if not synthesizer_config.reference_audio_path:
            raise ValueError("reference_audio_path is required for CambaiVertexSynthesizer")
        if not os.path.exists(synthesizer_config.reference_audio_path):
            raise ValueError(f"Reference audio file not found: {synthesizer_config.reference_audio_path}")
        
        # Set up authentication
        if synthesizer_config.credentials_path:
            if not os.path.exists(synthesizer_config.credentials_path):
                raise ValueError(f"Credentials file not found: {synthesizer_config.credentials_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = synthesizer_config.credentials_path
        
        # Store initialization flag
        self._vertex_ai_initialized = False
        
        # Load and encode reference audio
        try:
            with open(synthesizer_config.reference_audio_path, "rb") as f:
                self.reference_audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            logger.info(f"Loaded reference audio from {synthesizer_config.reference_audio_path}")
        except Exception as e:
            raise CambaiVertexSynthesizerException(f"Failed to load reference audio: {str(e)}")

    def _ensure_vertex_ai_initialized(self):
        """Initialize Vertex AI if not already initialized"""
        if not self._vertex_ai_initialized:
            try:
                aiplatform.init(
                    project=self.synthesizer_config.project_id,
                    location=self.synthesizer_config.location
                )
                logger.info(f"Initialized Vertex AI with project {self.synthesizer_config.project_id} in {self.synthesizer_config.location}")
                self._vertex_ai_initialized = True
            except Exception as e:
                raise CambaiVertexSynthesizerException(f"Failed to initialize Vertex AI: {str(e)}")

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: CambaiVertexSynthesizerConfig) -> str:
        """Create unique identifier for voice caching"""
        return ":".join([
            "cambai_vertex",
            synthesizer_config.project_id,
            synthesizer_config.endpoint_id,
            synthesizer_config.language,
            os.path.basename(synthesizer_config.reference_audio_path),
            synthesizer_config.audio_encoding.value,
            str(synthesizer_config.sampling_rate)
        ])

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        """Create speech using MARS7 via Vertex AI"""
        
        # Track character usage
        self.total_chars += len(message.text)
        
        try:
            # Prepare prediction payload
            instances = {
                "text": message.text,
                "audio_ref": self.reference_audio_base64,
                "language": self.synthesizer_config.language
            }
            
            # Add reference text if provided
            if self.synthesizer_config.reference_text:
                instances["ref_text"] = self.synthesizer_config.reference_text
            
            logger.debug(f"Synthesizing text: {message.text[:50]}...")
            
            # Make prediction
            audio_bytes = await self._predict_audio(instances)
            
            # Convert FLAC to required format
            converted_audio = await self._convert_audio_format(audio_bytes)
            
            # Create chunk generator
            async def chunk_generator():
                for i in range(0, len(converted_audio), chunk_size):
                    if i + chunk_size > len(converted_audio):
                        yield SynthesisResult.ChunkResult(converted_audio[i:], True)
                    else:
                        yield SynthesisResult.ChunkResult(converted_audio[i:i + chunk_size], False)
            
            # Create message cutoff function
            def get_message_up_to(seconds: Optional[float]) -> str:
                return BaseSynthesizer.get_message_cutoff_from_total_response_length(
                    self.synthesizer_config, message, seconds, len(converted_audio)
                )
            
            return SynthesisResult(
                chunk_generator=chunk_generator(),
                get_message_up_to=get_message_up_to
            )
            
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {str(e)}")
            raise CambaiVertexSynthesizerException(f"Speech synthesis failed: {str(e)}")

    async def _predict_audio(self, instances: dict) -> bytes:
        """Make prediction request to MARS7 model"""
        try:
            # Ensure Vertex AI is initialized
            self._ensure_vertex_ai_initialized()
            
            endpoint = aiplatform.Endpoint(endpoint_name=self.synthesizer_config.endpoint_id)
            
            # Prepare request data
            data = {"instances": [instances]}
            
            response = await endpoint.predict_async(instances=[instances])
            
            # Extract audio from response
            audio_base64 = response.predictions[0]
            audio_bytes = base64.b64decode(audio_base64)
            
            logger.debug(f"Received {len(audio_bytes)} bytes of audio from MARS7")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"MARS7 prediction failed: {str(e)}")
            raise CambaiVertexSynthesizerException(f"MARS7 prediction failed: {str(e)}")

    async def _convert_audio_format(self, flac_audio_bytes: bytes) -> bytes:
        """Convert FLAC audio to required format and sample rate"""
        try:
            # Read FLAC data from BytesIO
            flac_buffer = io.BytesIO(flac_audio_bytes)
            audio_data, original_sample_rate = sf.read(flac_buffer)
            
            # Write to WAV BytesIO
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, original_sample_rate, format='WAV')
            wav_buffer.seek(0)
            
            # Convert to required format using vocode's utility
            converted_audio = convert_wav(
                wav_buffer,
                output_sample_rate=self.synthesizer_config.sampling_rate,
                output_encoding=self.synthesizer_config.audio_encoding,
            )
            
            logger.debug(f"Converted audio: {len(converted_audio)} bytes at {self.synthesizer_config.sampling_rate}Hz")
            return converted_audio
                    
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            raise CambaiVertexSynthesizerException(f"Audio conversion failed: {str(e)}")
            