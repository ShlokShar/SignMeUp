# SignMeUp - The ASL Subtitle

SignMeUp bridges the gap for deaf and hard-of-hearing viewers by turning English captions into ASL subtitles, making stories, lessons, and conversations truly accessible to everyone.

## What it does
A user begins by uploading an MP4 video, after which SignMeUp extracts the audio and transcribes it into text using OpenAI’s Whisper model. The transcribed text is then passed through our an English to ASL translation pipeline, which reformats the English into a sequence of ASL signs. Each word in the sequence is matched against entries in the ASL-LEX database to retrieve corresponding sign images. Finally, these images are overlaid on top of the video, producing ASL subtitles that play alongside the original content.

## Built with
- Flask
- PyTorch
- TailwindCSS
- Whisper
- Stanza
- ASL-LEX

## Hackathon Submission
Submitted to Hello World 2025 – Winner Gold Track (3rd Place) 🥉
