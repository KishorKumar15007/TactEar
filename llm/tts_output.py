class MockSpeechOutput:
    def __init__(self): self.spoken=[]; self.stopped=False
    def speak(self,text): self.spoken.append(text); self.stopped=False
    def stop(self): self.stopped=True
class Pyttsx3SpeechOutput:
    def __init__(self):
        import pyttsx3; self.engine=pyttsx3.init()
    def speak(self,text): self.engine.say(text); self.engine.runAndWait()
    def stop(self): self.engine.stop()
