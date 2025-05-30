# Step 1: Define your FAQ data
faq_data = {
    "what is codealpha": "CodeAlpha is a platform offering internships and projects in tech domains like AI, web dev, and more.",
    "how long is the internship": "The internship lasts for 4 weeks and includes 3 tasks.",
    "how to submit my project": "Submit your GitHub repo link via the submission form in your WhatsApp group.",
    "do we get certificate": "Yes, you will receive a certificate after completing all 3 tasks.",
    "how many tasks to complete": "You must complete at least 3 out of the 4 tasks to get your certificate.",
    "where to post project update": "Post your updates and demo video on LinkedIn tagging @CodeAlpha.",
    "what is the duration of each task": "Each task should be completed within 5 days of assignment.",
    "can I choose any 3 tasks": "Yes, you can choose any 3 out of the 4 provided tasks.",
    "is there any registration fee": "No, the internship at CodeAlpha is completely free.",
    "can I use my own project idea": "No, you must complete the tasks assigned by CodeAlpha.",
    "do I need to upload on github": "Yes, upload your full source code on GitHub in a repo named CodeAlpha_ProjectName.",
    "do I need to make a video": "Yes, make a short video explaining your project and post it on LinkedIn.",
    "how to contact support": "You can contact support via services@codealpha.tech or through the WhatsApp group.",
    "can I do this internship remotely": "Yes, the internship is completely remote.",
    "will there be a final evaluation": "No formal evaluation, but incomplete tasks may affect your certificate.",
    "can I share my project on LinkedIn": "Yes, in fact, it's mandatory to share it and tag @CodeAlpha."
}
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract questions and answers
questions = list(faq_data.keys())
answers = list(faq_data.values())

# Convert text questions into numerical vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)
# Step 3: Define chatbot response function
def chatbot(user_input):
    user_vec = vectorizer.transform([user_input])  # Convert user input into vector
    similarities = cosine_similarity(user_vec, question_vectors)  # Compare to FAQs
    best_match = similarities.argmax()  # Find the most similar question
    if similarities[0][best_match] < 0.3:
        return "Sorry, I don't know the answer to that yet."
    return answers[best_match]
# Step 4: Chat loop
print("🤖 CodeAlpha Chatbot: Ask me anything about the internship! (type 'exit' to quit)\n")

while True:
    user_input = input("You: ").lower()
    if user_input == 'exit':
        print("Bot: Goodbye! All the best for your internship. 🎓")
        break
    response = chatbot(user_input)
    print("Bot:", response)