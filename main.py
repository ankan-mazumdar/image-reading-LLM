import streamlit as st
from PIL import Image
import base64
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

def encode_image(upload_file):
    image_bytes = upload_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return base64_image

def gen_response(api_key, base64_image, input):
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o")
    response = llm.invoke(
        [
            AIMessage(
                content="You are a useful & intelligent bot who is very good at image reading related OCR tasks to get insights from images uploaded by the user."
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": input},
                    {"type": "image_url",
                     "image_url": {
                         "url": "data:image/jpg;base64," + base64_image,
                         "detail": "auto"
                     }
                     }
                ]
            )
        ]
    )
    return response.content

def main():
    st.title("OBJECT DETECTION & ANALYSIS APP")

    api_key = st.text_input("Enter your OpenAI API key", type="password")
    upload_file = st.file_uploader("Upload your image here", type=["jpg"])
    
    if upload_file is not None:
        image = Image.open(upload_file)
        st.image(image, caption="Your image", use_column_width=True)
        st.success("Image uploaded successfully")
        
        base64_image = encode_image(upload_file)
        input = st.text_area("Ask your question here")
        
        if st.button("Submit") and api_key:
            response = gen_response(api_key, base64_image, input)
            st.write(response)
        elif not api_key:
            st.warning("Please enter your API key to proceed.")

if __name__ == "__main__":
    main()
