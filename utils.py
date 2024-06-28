from pandasai.responses.response_parser import ResponseParser
from pandasai.llm.openai import OpenAI
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import logging
import json
import glob
from flask import jsonify
import base64

load_dotenv()
openai_api_key = os.getenv("openai_api_key")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


class PandasDataFrame(ResponseParser):

    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        # Returns Pandas Dataframe instead of SmartDataFrame
        return result["value"]
    
llm = OpenAI(api_token=openai_api_key)


def load_df(dataset):
    try:#
        folder_path = 'diabetes/'
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        dataframes = []
        
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            dataframes.append(df)
        df= pd.concat(dataframes, ignore_index=True)

        return df
    except Exception as e:
        logging.error(f'Error loading dataset: {e}')

def load_amazon():

    df_amazon=pd.read_csv('amazon/Amazon Sale Report (1).csv',low_memory=False)
    return df_amazon

    

def serialize_smart_dataframe(data):
    def handle_nan(value):
        if isinstance(value,float) and pd.isna(value):
            return 'None'
        elif isinstance(value,int)and pd.isna(value):
            return 'None'
        elif isinstance(value,int)and pd.isna(value):
            return 'None'
        return value

    if isinstance(data, pd.DataFrame):
        print('dataframe')
        serialized_data = data.applymap(handle_nan).to_dict(orient='records')
        return jsonify({'table':serialized_data})
    
    elif isinstance(data, pd.Series):
        print('series')
        serialized_data = data.to_dict()
        return jsonify({'text': serialized_data})
    else:
        serialized_data = json.dumps(data, ensure_ascii=False)
        return jsonify({'text':serialized_data})

def process_question(df,prompt):
    try:
        response = df.chat(prompt)  
        return response
    except Exception as e:
        logging.error(f'Error processing question: {e}')
        raise

def get_latest_encoded_image():
    try:
        current_images = glob.glob("plots/*.png")

        # Get the most recent image
        most_recent_image = max(current_images, key=os.path.getmtime)
        with open(most_recent_image, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")

        # Remove older images
        for image in current_images:
            if image != most_recent_image:
                os.remove(image)

        return image_data
    except Exception as e:
        logging.error(f'Error processing images: {e}')
        return None