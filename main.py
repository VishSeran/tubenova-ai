import gradio as gr
from app import summary_generate, chat_with_llm, ingest_video

def gradio_interface():
    
    with gr.Blocks as interface:
        gr.Markdown(
            "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
        )
        
        #input video url
        video_url = gr.Textbox(label="Youtube video URL",placeholder="Enter Youtube Video URL here")
        
        #output summar and answer
        summary_output = gr.Textbox(label="Video Summary", lines=5)
        question_input = gr.Textbox(label="Ask a question about the video", placeholder="Ask a question")
        answer_output = gr.Textbox(label="Asnwer to your question", lines=5)
        
        # Buttons for selecting functionalities after fetching transcript
        summarize_btn = gr.Button("Summarize Video")
        question_btn = gr.Button("Ask a Question")
        
        transcript_sts = gr.Textbox(label="Transcript Status", interactive=False)
        
        #setup button actions
        summarize_btn.click(fn=summary_generate,
                            inputs=video_url,
                            outputs=summary_output)
        
        question_btn.click(
            fn=(ingest_video,chat_with_llm),
            inputs=[(video_url),(video_url,question_input)],
            outputs=answer_output
        )