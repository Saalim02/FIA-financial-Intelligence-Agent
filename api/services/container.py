from api.services.dataset_service import DatasetService
from api.services.query_service import QueryService
from api.services.voice_service import VoiceService


dataset_service = DatasetService()
query_service = QueryService(dataset_service)
voice_service = VoiceService()
