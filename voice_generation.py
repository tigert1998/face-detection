import requests
import threading

import ormsgpack
import uuid
import playsound

from schema import ServeTTSRequest


class VoiceGeneration:
    def __init__(self, url):
        self.url = url
        normalize = True
        self.format = "wav"
        max_new_tokens = 0
        chunk_length = 200
        top_p = 0.7
        self.streaming = False

        self.data = {
            # "references": [
            #     ServeReferenceAudio(audio=ref_audio, text=ref_text)
            #     for ref_text, ref_audio in zip(ref_texts, byte_audios)
            # ],
            # "reference_id": idstr,
            "normalize": normalize,
            "format": self.format,
            "max_new_tokens": max_new_tokens,
            "chunk_length": chunk_length,
            "top_p": top_p,
            # "repetition_penalty": repetition_penalty,
            # "temperature": temperature,
            "streaming": self.streaming,
            # "use_memory_cache": use_memory_cache,
            # "seed": seed,
        }

    def generate(self, text, folder_path):
        self.data["text"] = text
        pydantic_data = ServeTTSRequest(**self.data)
        response = requests.post(
            self.url,
            data=ormsgpack.packb(
                pydantic_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC
            ),
            stream=self.streaming,
            headers={
                # "authorization": "Bearer YOUR_API_KEY",
                "content-type": "application/msgpack",
            },
        )

        assert not self.streaming
        assert response.status_code == 200
        audio_path = f"{folder_path}/{uuid.uuid1()}.{self.format}"
        audio_content = response.content
        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_content)
        return audio_path


class VoiceQueue:
    def __init__(self, voice_generation: VoiceGeneration):
        self.stop_flag = False
        self.queue = []
        self.condition = threading.Condition()
        self.voice_generation = voice_generation

        self.thread = threading.Thread(target=self.play, args=())
        self.thread.start()

    def play(self):
        while not self.stop_flag:
            with self.condition:
                while len(self.queue) == 0 and not self.stop_flag:
                    self.condition.wait()
                if self.stop_flag:
                    return
                text = self.queue[0]
                self.queue = self.queue[1:]

            path = self.voice_generation.generate(text, "temp")
            playsound.playsound(path, block=True)

    def stop(self):
        with self.condition:
            self.stop_flag = True
            self.condition.notify_all()
        self.thread.join()

    def add_task(self, text):
        with self.condition:
            self.queue.append(text)
            self.condition.notify_all()
