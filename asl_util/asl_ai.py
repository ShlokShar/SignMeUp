import stanza
import os
import re
import torch
from typing import List, Dict
import logging
from num2words import num2words
import tempfile

# Try multiple whisper alternatives in order of preference
WHISPER_BACKEND = None

# Option 1: Try faster-whisper (more reliable, fewer dependencies)
try:
    from faster_whisper import WhisperModel

    WHISPER_BACKEND = "faster-whisper"
    print("Using faster-whisper backend")
except ImportError:
    pass

# Option 2: Try whisper-timestamped 
if WHISPER_BACKEND is None:
    try:
        import whisper_timestamped as whisper

        WHISPER_BACKEND = "timestamped"
        print("Using whisper-timestamped backend")
    except ImportError:
        pass

# Option 3: Fall back to regular whisper
if WHISPER_BACKEND is None:
    try:
        import whisper

        WHISPER_BACKEND = "original"
        print("Using original whisper backend (may need FFmpeg)")
    except ImportError:
        print("No whisper backend available!")
        exit(1)

# Suppress logging
logging.getLogger('stanza').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)


class EnglishToASLConverter:
    def __init__(self):
        self.gpu_armed = torch.cuda.is_available()
        self.nlp = stanza.Pipeline(
            processors='tokenize,pos,lemma,depparse',
            lang='en',
            use_gpu=self.gpu_armed
        )
        self.blacklist = {
            ('is', 'be'), ('the', 'the'), ('of', 'of'), ('are', 'be'),
            ('by', 'by'), (',', ','), (';', ';'), (':', ':'), ('a', 'a'), ('an', 'a')
        }

    def text_to_asl(self, text: str) -> List[Dict]:
        print(f"Processing: {text}")
        doc = self.nlp(text)
        asl_translation = []
        for sentence in doc.sentences:
            sentence_translation = self._translate_sentence(sentence)
            asl_translation.extend(sentence_translation)
        return asl_translation

    def _translate_sentence(self, sentence) -> List[Dict]:
        ordered_words = self._get_ordered_words(sentence)
        asl_sequence = self._convert_to_asl_sequence(ordered_words)
        return asl_sequence

    def _get_ordered_words(self, sentence) -> List[Dict]:
        words = []
        for token in sentence.tokens:
            for word in token.words:
                word_dict = {
                    'index': word.id,
                    'governor': word.head,
                    'text': word.text.lower(),
                    'lemma': word.lemma.lower(),
                    'upos': word.upos,
                    'dependency_relation': word.deprel
                }
                words.append(word_dict)
        words.sort(key=lambda x: (x['governor'], x['index']))
        return words

    def _convert_to_asl_sequence(self, words: List[Dict]) -> List[Dict]:
        asl_signs = []
        sentence_tone = ""
        for word in words:
            if (word['text'], word['lemma']) in self.blacklist:
                continue
            # if word['upos'] == 'PUNCT':
            #     if word['lemma'] == "?":
            #         sentence_tone = "question"
            #     elif word['lemma'] == "!":
            #         sentence_tone = "exclamation"
            #     continue
            if word['upos'] in ['SYM', 'X', 'PART']:
                continue
            sign_info = self._process_word_by_pos(word)
            if sign_info:
                if isinstance(sign_info, list):
                    asl_signs.extend(sign_info)
                else:
                    asl_signs.append(sign_info)
        if sentence_tone:
            asl_signs.append({
                'text': f"[{sentence_tone}]",
                'lemma': '',
                'type': 'tone',
                'instruction': f"Use {sentence_tone} facial expression"
            })
        return asl_signs

    def _process_word_by_pos(self, word: Dict) -> List[Dict] or Dict or None:
        if word['upos'] == 'PROPN':
            return self._create_fingerspell_sequence(word['text'])
        elif word['upos'] == 'NUM':
            return self._create_number_sign(word['text'])
        elif word['upos'] in ['AUX', 'DET', 'ADP']:
            return None
        elif word['upos'] in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'INTJ', 'CCONJ', 'SCONJ']:
            if word['upos'] == 'PRON' and word['dependency_relation'] == 'nsubj':
                return None
            if word['upos'] == 'SCONJ' and word['lemma'] == 'that':
                return None
            return {
                'text': word['text'],
                'lemma': word['lemma'],
                'type': 'sign',
                'pos': word['upos'],
                'instruction': word['lemma']
            }
        return None

    def _create_fingerspell_sequence(self, word: str) -> List[Dict]:
        fingerspell_sequence = []
        for letter in word.lower():
            if letter.isalpha():
                fingerspell_sequence.append({
                    'text': letter,
                    'lemma': letter,
                    'type': 'fingerspell',
                    'instruction': f"Fingerspell letter '{letter.upper()}'"
                })
        return fingerspell_sequence

    def _create_number_sign(self, number: str) -> Dict:
        number = num2words(int(number))
        return {
            'text': number,
            'lemma': number,
            'type': 'number',
            'instruction': number
        }

    def display_asl_sequence(self, asl_sequence: List[Dict]):
        final_str = ''
        for i, sign in enumerate(asl_sequence, 1):
            final_str += sign['instruction']
            final_str += ' '
            if sign['type'] == 'fingerspell':
                continue
        print(final_str)


class UniversalWhisperTranscriber:
    def __init__(self, model_size="tiny", use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.backend = WHISPER_BACKEND

        if self.backend == "faster-whisper":
            # Faster-whisper is more reliable and doesn't need FFmpeg
            self.model = WhisperModel(model_size, device=self.device, compute_type="float16" if use_gpu else "int8")

        elif self.backend == "timestamped":
            # Whisper-timestamped handles dependencies better
            self.model = whisper.load_model(model_size, device=self.device)

        else:
            # Original whisper (may fail without FFmpeg)
            self.model = whisper.load_model(model_size, device=self.device)

    def transcribe_optimized(self, audio_file, **kwargs):
        print(f"Transcribing {audio_file} with {self.backend}...")

        try:
            if self.backend == "faster-whisper":
                segments, info = self.model.transcribe(audio_file, language="en")
                transcript = " ".join([segment.text for segment in segments])

            elif self.backend == "timestamped":
                result = whisper.transcribe(self.model, audio_file, language="en")
                transcript = result["text"]

            else:
                # Original whisper
                result = self.model.transcribe(audio_file, language="en")
                transcript = result["text"]

            return transcript.strip()

        except Exception as e:
            print(f"Transcription failed with {self.backend}: {e}")
            return None


def transcribe_english_only(mp4_file_path):
    if not os.path.exists(mp4_file_path):
        print(f"File not found: {mp4_file_path}")
        return None

    print(f"Processing: {mp4_file_path}")

    try:
        transcript = transcriber.transcribe_optimized(mp4_file_path)

        if transcript is None:
            print("Transcription failed")
            return None

        sentences = re.split(r'(?<=[.?!])\s+', transcript)
        output_file = f"{os.path.splitext(mp4_file_path)[0]}_transcript.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcript)

        print(f"Transcript saved to: {output_file}")
        return sentences

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def main(file_path):
    sen = transcribe_english_only(file_path)
    if sen is None:
        return

    converter = EnglishToASLConverter()
    for sentence in sen:
        if sentence.strip():
            print(f"\nInput: {sentence}")
            asl_sequence = converter.text_to_asl(sentence)
            out = converter.display_asl_sequence(asl_sequence)
            #converter.display_asl_sequence(asl_sequence)
            input("Press Enter to continue...")


# Initialize transcriber
print("Initializing Universal Whisper...")
transcriber = UniversalWhisperTranscriber(model_size="tiny", use_gpu=False)

if __name__ == '__main__':
    main("C:\\Users\\Ari\\Downloads\\videoplayback (1).mp4")
