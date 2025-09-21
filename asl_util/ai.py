import stanza
import os
import re
import torch
from typing import List, Dict
import logging
from num2words import num2words
import tempfile
import platform
from pathlib import Path

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

# Detect platform for platform-specific optimizations
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

print(f"Running on: {platform.system()} {platform.release()}")

# Mac-specific optimizations
if IS_MAC:
    # Use Metal Performance Shaders if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) available - using GPU acceleration on Mac")
        DEFAULT_DEVICE = "mps"
    else:
        DEFAULT_DEVICE = "cpu"
        print("Using CPU on Mac (MPS not available)")
else:
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Suppress logging
logging.getLogger('stanza').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

class EnglishToASLConverter:
    def __init__(self):
        # Cross-platform GPU detection
        if IS_MAC and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.gpu_armed = True
            self.device = "mps"
        elif torch.cuda.is_available():
            self.gpu_armed = True
            self.device = "cuda"
        else:
            self.gpu_armed = False
            self.device = "cpu"
            
        print(f"ASL Converter using device: {self.device}")
        
        # Initialize Stanza with proper device handling
        try:
            self.nlp = stanza.Pipeline(
                processors='tokenize,pos,lemma,depparse',
                lang='en',
                use_gpu=self.gpu_armed and self.device == "cuda"  # Stanza only supports CUDA GPU
            )
        except Exception as e:
            print(f"Warning: GPU initialization failed, falling back to CPU: {e}")
            self.nlp = stanza.Pipeline(
                processors='tokenize,pos,lemma,depparse',
                lang='en',
                use_gpu=False
            )
            
        self.blacklist = {
            ('is', 'be'), ('the', 'the'), ('of', 'of'), ('are', 'be'),
            ('by', 'by'), (',', ','), (';', ';'), (':', ':'), ('a', 'a'), ('an', 'a')
        }

    def text_to_asl(self, text: str) -> List[Dict]:
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
                    'instruction': letter.upper()
                })
        return fingerspell_sequence

    def _create_number_sign(self, number: str) -> Dict:
        """Handle number input with better error handling"""
        try:
            # Clean the number string - remove common punctuation
            clean_number = number.strip('.,!?;:')
            
            # Handle common cases
            if clean_number.isdigit():
                # Simple integer
                number_word = num2words(int(clean_number))
            elif '.' in clean_number and clean_number.replace('.', '').isdigit():
                # Decimal number - convert to float first
                number_word = num2words(float(clean_number))
            else:
                # If it's not a clear number, just use the original text
                number_word = clean_number
                
            return {
                'text': number,
                'lemma': number_word,
                'type': 'number',
                'instruction': number_word
            }
            
        except (ValueError, TypeError) as e:
            # If conversion fails, treat it as regular text
            print(f"Warning: Could not convert '{number}' to number, treating as text")
            return {
                'text': number,
                'lemma': number,
                'type': 'sign',
                'instruction': number
            }

    def display_asl_sequence(self, asl_sequence: List[Dict]):
        final_str = ''
        for i, sign in enumerate(asl_sequence, 1):
            final_str += sign['instruction']
            final_str += ' '
            if sign['type'] == 'fingerspell':
                continue
        return final_str

class UniversalWhisperTranscriber:
    def __init__(self, model_size="tiny", use_gpu=None):
        # Smart device selection based on platform
        if use_gpu is None:
            if IS_MAC:
                # Mac: prefer MPS if available, otherwise CPU
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                    use_gpu = True
                else:
                    self.device = "cpu"
                    use_gpu = False
            else:
                # Windows/Linux: use CUDA if available
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                use_gpu = torch.cuda.is_available()
        else:
            if use_gpu and IS_MAC and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            elif use_gpu and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                use_gpu = False
        
        print(f"Whisper using device: {self.device}")
        self.backend = WHISPER_BACKEND
        
        try:
            if self.backend == "faster-whisper":
                # Faster-whisper device mapping
                if self.device == "mps":
                    # faster-whisper doesn't support MPS directly, use CPU
                    device = "cpu"
                    compute_type = "int8"
                elif self.device == "cuda":
                    device = "cuda"
                    compute_type = "float16" if use_gpu else "int8"
                else:
                    device = "cpu"
                    compute_type = "int8"
                    
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            
            elif self.backend == "timestamped":
                # Handle MPS for whisper-timestamped
                if self.device == "mps":
                    # Some whisper versions don't support MPS directly
                    try:
                        self.model = whisper.load_model(model_size, device=self.device)
                    except:
                        print("MPS not supported by whisper-timestamped, falling back to CPU")
                        self.device = "cpu"
                        self.model = whisper.load_model(model_size, device=self.device)
                else:
                    self.model = whisper.load_model(model_size, device=self.device)
                
            else:
                # Original whisper with MPS support
                if self.device == "mps":
                    try:
                        self.model = whisper.load_model(model_size, device=self.device)
                    except:
                        print("MPS not supported by original whisper, falling back to CPU")
                        self.device = "cpu"
                        self.model = whisper.load_model(model_size, device=self.device)
                else:
                    self.model = whisper.load_model(model_size, device=self.device)
                    
        except Exception as e:
            print(f"Error initializing Whisper model: {e}")
            print("Falling back to CPU...")
            self.device = "cpu"
            
            if self.backend == "faster-whisper":
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            elif self.backend == "timestamped":
                self.model = whisper.load_model(model_size, device="cpu")
            else:
                self.model = whisper.load_model(model_size, device="cpu")

    def transcribe_optimized(self, audio_file, **kwargs):
        # Cross-platform path handling
        audio_file = str(Path(audio_file).resolve())
        print(f"Transcribing {audio_file} with {self.backend} on {self.device}...")
        
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
    # Cross-platform path handling
    mp4_file_path = Path(mp4_file_path).resolve()
    
    if not mp4_file_path.exists():
        print(f"File not found: {mp4_file_path}")
        return None

    print(f"Processing: {mp4_file_path}")
    
    try:
        transcript = transcriber.transcribe_optimized(str(mp4_file_path))
        
        if transcript is None:
            print("Transcription failed")
            return None
            
        sentences = re.split(r'(?<=[.?!])\s+', transcript)
        
        # Cross-platform output file path
        output_file = mp4_file_path.with_suffix('').with_suffix('.txt').with_name(
            mp4_file_path.stem + "_transcript.txt"
        )
        
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
    FINAL_STR = ''
    for sentence in sen:
        if sentence.strip():
            asl_sequence = converter.text_to_asl(sentence)
            out = converter.display_asl_sequence(asl_sequence)
            FINAL_STR += out
            #converter.display_asl_sequence(asl_sequence)
    return FINAL_STR

# Initialize transcriber with smart device selection
print("Initializing Universal Whisper...")
transcriber = UniversalWhisperTranscriber(model_size="tiny", use_gpu=None)

if __name__ == '__main__':
    # Cross-platform example path
    if IS_MAC:
        # Mac example path
        example_path = os.path.expanduser("~/Downloads/videoplayback.mp4")
    elif IS_WINDOWS:
        # Windows example path
        example_path = r"C:\Users\{}\Downloads\videoplayback.mp4".format(os.getenv('USERNAME', 'User'))
    else:
        # Linux example path
        example_path = os.path.expanduser("~/Downloads/videoplayback.mp4")
    
    # Check if example file exists, otherwise prompt user
    if not os.path.exists(example_path):
        print(f"Example file not found at: {example_path}")
        print("Please provide a valid video file path:")
        file_input = input("Enter path to video file: ").strip().strip('"\'')
        if file_input:
            example_path = file_input
        else:
            print("No file provided, exiting.")
            exit(1)
    
    r = main(example_path)
    print("ASL Translation:", r)