import os
import google.generativeai as genai

# Set API key for Google Generative AI
os.environ["API_KEY"] = "AIzaSyDdn2-vk5kqKTZZPfViW_9tSaM2ab7PaK8"  # Replace with your actual API key
genai.configure(api_key=os.environ["API_KEY"])

def generate_text(prompt):
    """Generates a response using Google Generative AI with an identity reinforcement."""
    # Custom prompt to reinforce the identity as Banglabot
    custom_prompt = (
        f"আপনি একজন চ্যাটবট যার নাম Banglabot। "
        "আপনার কাজ হল ব্যবহারকারীদের প্রশ্নের উত্তর দেওয়া এবং তাদের সাথে কথোপকথন করা। "
        "আপনি গুগল দ্বারা প্রশিক্ষিত নন; আপনি শুধুমাত্র Banglabot। "
        "দয়া করে বাংলা ভাষায় উত্তর দিন। "
        f"ব্যবহারকারীর প্রশ্ন: {prompt}"
    )

    # Create generation configuration
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Create model object
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Start chat session and send message
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(custom_prompt)
    
    return response.text
