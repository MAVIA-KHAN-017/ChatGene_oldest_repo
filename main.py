import pandas as pd
from pandasai import SmartDataframe, SmartDatalake
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flasgger import Swagger
from flask_cors import CORS
import sys
import os
import matplotlib
from utils import llm,PandasDataFrame,serialize_smart_dataframe,process_question,get_latest_encoded_image,load_amazon
matplotlib.use('agg')  # Use the 'agg' backend

import logging

app = Flask(__name__)
CORS(app) 
swagger = Swagger(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
df=load_amazon()

df = SmartDatalake(
    [df],
    config={
        "llm": llm,
        "is_chat_model": True,
        "enable_cache": False,
        "max_retries": 5,
        "use_error_correction_framework": True,
        "verbose": False,
        "sample_head": df,
        "enforce_privacy": False,
        "save_charts": True,
        "save_charts_path": 'plots',
        "response_parser": PandasDataFrame
    },
)

@app.route('/ask/<prompt>', methods=['GET'])
def ask_question(prompt:str):
    """Endpoint to ask a question and get the response and graph."""
    try:
        assert isinstance(prompt, str), "Prompt must be a string"        
        response = process_question(df,prompt)
        if isinstance(response, (pd.DataFrame, pd.Series)):
            response_data = serialize_smart_dataframe(response)
            return response_data
        
        elif response is None:
            image_data = get_latest_encoded_image()
            if image_data:
                return jsonify({"image": image_data})
            else:
                return jsonify({"error": "No data or images available"}), 404
        else:
            return jsonify({'text':response})


    except ValueError as ve:
        logging.error(f'Value Error: {ve}')
        return jsonify({"error": "Invalid input value"}), 400  # Bad Request
    
    except KeyError as ke:
        logging.error(f'Key Error: {ke}')
        return jsonify({"error": "Key not found"}), 404  # Not Found

    except Exception as e:
        logging.error(f'Error: {e}')
        return jsonify({"error": "Internal Server Error"}), 500



if __name__ == '__main__':

    app.run()
