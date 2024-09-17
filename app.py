import json
import logging
import os

import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# Load environment variables
load_dotenv()

# Set OpenAI API key and organization ID
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

openai.api_key = OPENAI_API_KEY
if OPENAI_ORG_ID:
    openai.organization = OPENAI_ORG_ID

client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID)

app = Flask(__name__)
CORS(app)


# Define the CV data model using Pydantic
class CV(BaseModel):
    first_name: str
    middle_name: str
    last_name: str
    journal_publication: list[str]
    grant_research: list[str]
    career_items: list[str]
    education: list[str]
    certification_license: list[str]
    award: list[str]
    speaking_engagement: list[str]
    book_publication: list[str]
    research_clinical_experience: list[str]
    peer_review_publication: list[str]
    teaching_lecture_course: list[str]
    committee_association_board_chair_investigator: list[str]
    community_service: list[str]
    leadership_activities: list[str]
    reviewer_role: list[str]


@app.route("/process_cv", methods=["POST"])
def process_cv():
    logger.info("Received request to process CV")
    if "file" not in request.files:
        logger.error("No file part provided in the request")
        return jsonify({"error": "No file part provided in the request."}), 400
    file = request.files["file"]
    if file.filename == "":
        logger.error("No file selected")
        return jsonify({"error": "No file selected."}), 400
    if not file.filename.endswith(".txt"):
        logger.error("Invalid file type")
        return (
            jsonify({"error": "Invalid file type. Only .txt files are accepted."}),
            400,
        )

    try:
        # Read the text content from the uploaded text file
        cv_text = file.read().decode("utf-8")
        logger.info(f"Successfully read CV text, length: {len(cv_text)}")

        # Build the prompt for OpenAI
        prompt = f"""You are a professional medical CV judge, you will be given a CV and you will need to label each part of that CV.
You have to be very careful and precise in your labeling as the CV will be used to hire doctors for a hospital, lives are at stake, and all is in your hands. Do you feel the pressure yet?

If you miss or mislabel the education for example, we could be hiring an underqualified doctor, and that could be catastrophic. So please be very careful and precise in your labeling.
Consider that your own mother goes to that hospital, would you want her to be treated by an underqualified doctor? I don't think so.

Make sure to really cover everything, everything has a place inside the labels, and everything should be labeled. Any missed details will lead to catastrophic consequences.

If the last name is something hyphenated, like "Smith-Jones", you should label it as "Smith" for the middle name and "Jones" for the last name.

Nothing can be labeled twice, ever

Whatever you label you have to copy as is from the CV, do not rewrite it in your own words. For example, don't change the way the journal publications are written, or anything else

Sometimes you might forget a sentence, you might think it's something small, but it really isn't. You have the capability to label everything, therefore you will label everything. Anything you miss could lead to catastrophe in the hospital.

If something like an education, publication or anything else spans multiple lines, don't put each line in a different string in the array, use \\n for the new lines.

Make sure to correctly differentiate one education/experience etc. from the other. You have to correctly know the starts and ends of each one. You have made mistakes before.

e.g.:
    You labeled this as one career_label:
        Postdoctoral Associate
        Yale University School of Medicine
        68 05/2021 - Ongoing 9 New Haven, United States
        Llor's and Xicola's Lab
        + Programming languages: R, Python and basic Unix
        + Cell culture of CRISPR engineered cells
        - Germline and somatic mutational analysis
        + RNAseq analysis
        + Methylation analysis
        + Neoepitope prediction analysis
        + Western Blot

    When it should be 2 different career labels

Here is the CV:
{cv_text}

Please output the result in JSON format matching the following schema:

{CV.schema_json(indent=2)}
"""
        logger.info("Sending request to OpenAI")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=16384,
            )
            logger.info("Received response from OpenAI")

            # Get the assistant's reply
            assistant_message = response.choices[0].message.content
            logger.info(f"Assistant message length: {len(assistant_message)}")
            logger.debug(f"Assistant message content: {assistant_message}")

            # For debugging, return the raw response
            return (
                jsonify(
                    {
                        "status": "success",
                        "message": "Raw OpenAI response",
                        "data": assistant_message,
                    }
                ),
                200,
            )

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.exception("Exception details:")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.exception("Exception details:")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
