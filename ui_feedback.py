import gradio as gr
import torch
import requests
from PIL import Image
from io import BytesIO
import training_log  # import the log module

def run_feedback_ui(df, embeddings, agent, sbert_model):
    """Creates an interactive UI for recommendations using SBERT and RL."""
    
    # States to store the last recommended title and user embedding
    recommended_title_state = gr.State(value=None)
    last_user_embedding_state = gr.State(value=None)

    def recommend(user_synopsis):
        """Generates recommendations based on the user-entered synopsis."""
        if not user_synopsis.strip():
            return "Please enter a synopsis to get recommendations.", None, None, None

        # Embed user input using SBERT
        with torch.no_grad():
            user_embedding = torch.tensor(sbert_model.encode([user_synopsis]), dtype=torch.float32).squeeze(0)

        # Record the synopsis and embedding for evaluation
        training_log.training_attempts += 1
        training_log.sbert_synopsis_list.append(user_synopsis)
        training_log.sbert_embedding_list.append(user_embedding.numpy())

        # Use RL agent to select actions based on the embedding
        recommended_idx = agent.select_actions(user_embedding)
        agent.history.extend(recommended_idx)  # update history with recommended indices

        # Retrieve details for the first recommended item (as an example)
        rec = df.iloc[recommended_idx[0]]
        formatted_text = (
            f"### {rec['Title']}\n"
            f"**Genres:** {rec['Genres']}\n\n"
            f"**Synopsis:** {rec['Synopsis']}"
        )
        # Process cover image: convert URL to PIL image
        cover_url = rec['Medium_Picture'].replace("\\", "/")
        try:
            response = requests.get(cover_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            print("Error loading image from URL:", e)
            img = None

        return formatted_text, img, rec['Title'], user_embedding

    def update_model(feedback, recommended_title, user_embedding):
        """Trains the model based on user feedback."""
        if recommended_title is None or user_embedding is None:
            return "No recommendation to provide feedback on!", None

        # Find the recommended title in the dataframe
        recommended_idx_list = df.index[df['Title'] == recommended_title].tolist()
        if not recommended_idx_list:
            return "Error: Recommended title not found in dataset.", None

        recommended_idx = recommended_idx_list[0]
        reward = 1 if feedback == "Good" else -1  # Simple +1/-1 reward

        # Update the RL agent using the stored user embedding
        agent.train(user_embedding, recommended_idx, reward)
        return f"Feedback received for '{recommended_title}'! Get another recommendation.", None

    # Gradio UI
    with gr.Blocks() as ui:
        gr.Markdown("## Light Novel Recommendation System")
        gr.Markdown("Enter a synopsis to get personalized recommendations!")

        user_input = gr.Textbox(label="Enter a synopsis")
        recommend_button = gr.Button("Recommend")

        with gr.Row():
            with gr.Column(scale=1):
                cover_image = gr.Image(label="Cover Image", interactive=False)
            with gr.Column(scale=2):
                recommendation_text = gr.Markdown()

        # Define states for recommended title and last user embedding
        recommended_title_state = gr.State(value=None)
        last_user_embedding_state = gr.State(value=None)

        recommend_button.click(
            recommend,
            inputs=[user_input],
            outputs=[recommendation_text, cover_image, recommended_title_state, last_user_embedding_state]
        )

        feedback_message = gr.Textbox(label="Feedback Status", interactive=False)

        with gr.Row():
            feedback_good = gr.Button("Good")
            feedback_bad = gr.Button("Bad")

        feedback_good.click(
            lambda: "Good",
            outputs=[feedback_message]
        ).then(
            update_model,
            inputs=[feedback_message, recommended_title_state, last_user_embedding_state],
            outputs=[feedback_message, last_user_embedding_state]
        )

        feedback_bad.click(
            lambda: "Bad",
            outputs=[feedback_message]
        ).then(
            update_model,
            inputs=[feedback_message, recommended_title_state, last_user_embedding_state],
            outputs=[feedback_message, last_user_embedding_state]
        )

        ui.launch()
