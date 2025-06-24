import re
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
import os
import logging
import fitz  # PyMuPDF for PDF processing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def prepare_text_for_nlp(file_path, output_suffix):
    """
    Process a transcript file for NLP, saving the cleaned transcript as ready_for_nlp_<suffix>.txt.

    Args:
        file_path (str): Full path to the input file (.txt or .pdf)
        output_suffix (str): Suffix for the output file (e.g., 'current', 'prev1')
    """
    # Construct output file name
    output_file = f"ready_for_nlp_{output_suffix}.txt"

    # Function to sanitize filename for use in sentence IDs
    def sanitize_filename(filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        name = re.sub(r'[^a-zA-Z0-9\-]', '_', name)
        return name

    # Function to extract text from a PDF
    def extract_text_from_pdf(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text("text")
            doc.close()
            logging.info(f"Successfully extracted text from PDF: {file_path}")
            return text
        except Exception as e:
            logging.error(f"Failed to extract text from PDF {file_path}: {str(e)}")
            raise

    # Function to read text from a file (text or PDF)
    def read_file(file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                logging.info(f"Successfully read text file with UTF-8: {file_path}")
                return text
            except UnicodeDecodeError:
                fallback_encodings = ['windows-1252', 'latin-1', 'utf-16']
                for encoding in fallback_encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            text = f.read()
                        logging.info(f"Successfully read text file with fallback encoding {encoding}: {file_path}")
                        return text
                    except UnicodeDecodeError:
                        logging.warning(f"Failed to decode text file {file_path} with {encoding}")
                raise UnicodeDecodeError(f"Could not decode text file {file_path} with any supported encoding", b"", 0, 0, "")
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .txt or .pdf")

    # Function to clean and preprocess the transcript
    def clean_transcript(full_text, filename):
        # Generic start terms for transcript detection
        start_terms = ["Moderator", "Operator", "Host", "Conference Call Facilitator", "PRESENTATION", "Final Transcript"]

        # Generic regex patterns for noise detection
        noise_patterns = [
            r"©.*?\d{4}.*?",  # Copyright notices
            r".*?(Earnings Conference Call|Transcript of|Republished with permission|No part of this publication).*",  # Headers
            r".*? - \d+ -",  # Page numbers
            r".*?Event Date/Time:.*",  # Timestamps
            r".*?Transcription:.*",  # Metadata
            r".*?(Good (morning|afternoon|evening) and (thank you|welcome)|Before I hand over|With this, I hand over|Recording of this call|Thank you (everyone|all) for joining|In the interest of time|I (want|would like) to.*remind).*",  # Procedural content
        ]

        # Split the text into lines
        lines = full_text.split("\n")

        # Find the start of the transcript
        start_index = next(
            (i for i, line in enumerate(lines)
             if any(term in line.strip() for term in start_terms) or re.search(r"(welcome to|good (morning|afternoon|evening))", line.lower())),
            0  # Default to start if no match
        )
        transcript_lines = lines[start_index:]

        # Frequency-based noise detection for repetitive lines
        line_counts = Counter(transcript_lines)
        frequent_lines = [line for line, count in line_counts.items() if count > 2 and len(line.strip()) < 150]

        # Clean transcript lines by removing noise
        clean_lines = [
            line for line in transcript_lines
            if not any(re.search(pattern, line, re.IGNORECASE) for pattern in noise_patterns)
            and line.strip() not in frequent_lines
            and not line.strip().isdigit()
            and len(line.strip()) > 0
        ]

        # Group lines into speaker blocks and extract roles
        speaker_blocks = []
        current_speaker = None
        current_role = None
        current_text = []
        role_pattern = r"(CEO|Chief Executive Officer|Managing Director|CFO|Chief Financial Officer|Director|Head of|President|Vice President|Chairman|Analyst|Investor Relations)"

        for line in clean_lines:
            # Extract speaker name and role (handle various formats)
            match = re.match(r"(\w+.*?)?:?\s*(.*?)(?: - (.*?))?(?::.*)?$", line)
            if match and len(match.group(2).strip()) < 50 and all(word[0].isupper() for word in match.group(2).split() if word):
                speaker = match.group(2).strip()
                role = None
                if match.group(3):
                    role_match = re.search(role_pattern, match.group(3), re.IGNORECASE)
                    role = role_match.group(1) if role_match else None
                # Remove titles from speaker name
                speaker = re.sub(r"\s*(–|-)\s*(CEO|Chief Executive Officer|Managing Director|CFO|Chief Financial Officer|Director|Head of|President|Vice President|Chairman|Analyst|Investor Relations).*", "", speaker, flags=re.IGNORECASE).strip()
                if current_speaker:
                    speaker_blocks.append((current_speaker, current_role, " ".join(current_text)))
                current_speaker = speaker
                current_role = role
                current_text = [line[line.find(":", len(speaker)+2)+2:].strip() if line.find(":", len(speaker)+2) != -1 else line.strip()]
            else:
                current_text.append(line)
        if current_speaker:
            speaker_blocks.append((current_speaker, current_role, " ".join(current_text)))

        # Filter out procedural content dynamically
        procedural_patterns = r"(thank you (for joining|everyone|all)|hand over|remind.*participants|next question|bring this call to a close|recording of this call|please go ahead|now take.*questions)"
        relevant_blocks = [
            (speaker, role, text) for speaker, role, text in speaker_blocks
            if not re.search(procedural_patterns, text.lower(), re.IGNORECASE)
        ]

        # Clean and split each speaker's text into sentences
        final_output = []
        sanitized_filename = sanitize_filename(filename)
        sentence_idx = 1
        for speaker, role, text in relevant_blocks:
            # Remove bullet points, strange dashes, and normalize spaces
            cleaned_text = re.sub(r'^\s*[-•–]\s+', '', text, flags=re.MULTILINE)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            # Split into sentences
            sentences = sent_tokenize(cleaned_text)
            # Prefix each sentence with "Filename_Number | Speaker (Role):"
            for sentence in sentences:
                if sentence.strip():
                    role_suffix = f" ({role})" if role else ""
                    final_output.append(f"{sanitized_filename}_{sentence_idx} | {speaker}{role_suffix}: {sentence}")
                    sentence_idx += 1

        return "\n".join(final_output)

    # Process the transcript
    try:
        # Read text from file (PDF or text)
        full_text = read_file(file_path)
        # Clean and preprocess the transcript
        cleaned_transcript = clean_transcript(full_text, file_path)
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_transcript)
        logging.info(f"Transcript cleaned and saved to: {output_file}")
    except Exception as e:
        logging.error(f"Error processing transcript: {str(e)}")
        raise


if __name__ == "__main__":
  prepare_text_for_nlp("data/raw/INFY_Q1_July_22.pdf", "current")
  prepare_text_for_nlp("data/raw/INFY_Q2_October_21.pdf", "prev1")
  prepare_text_for_nlp("data/raw/INFY_Q3_January_22.pdf", "prev2")
  prepare_text_for_nlp("data/raw/INFY_Q4_April_22.pdf", "prev3")
