import os
from google.cloud import texttospeech
from constants import LANGUAGES, LANGUAGES_TO_USE
from translate import load_common_words
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/daniellongo/keys/lang-d5630e9c6c37.json"

CLIENT = texttospeech.TextToSpeechClient()


def main():
    already_processed = ["french", "english"]  # ["spanish", "latin", "korean"]
    for language in LANGUAGES_TO_USE:
        if language in already_processed:
            print("ALREADY READ SKIPPING", language)
            continue
        filename = "./translations/" + language + "_common-nouns.csv"
        words = load_common_words(filename)
        read_words(words, language)
    # read_word("hola", "spanish", "./audio/test4.mp3")


def read_words(words, language, save_filename_prefix=None):
    if save_filename_prefix is None:
        save_filename_prefix = "./audio/" + language + "/"
    else:
        save_filename_prefix = "./audio/" + language + "/" + save_filename_prefix + "-"
    for word in words:
        cur_filename = save_filename_prefix + word + ".mp3"
        read_word(word, language, cur_filename)


def read_word(word, language, save_filename):
    """Synthesizes speech from the input string of text or ssml.

    Saves as mp3 file

    Note: ssml must be well-formed according to:
        https://www.w3.org/TR/speech-synthesis/
    """

    # Instantiates a client
    print("word", word)
    time.sleep(1)

    language = LANGUAGES[language]
    # Set the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(text=word)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.types.VoiceSelectionParams(
        language_code=language,
        ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

    # Select the type of audio file you want returned
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = CLIENT.synthesize_speech(synthesis_input, voice, audio_config)

    # The response's audio_content is binary.
    with open(save_filename, 'wb') as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file', save_filename)


if __name__ == "__main__":
    main()
