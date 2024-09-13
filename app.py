from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from src.SubjectClassifier import logger
from src.SubjectClassifier.pipeline.prediction import Predictionpipeline
from src.SubjectClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.SubjectClassifier.pipeline.stage_02_prepare_model import Preparefintunemodelpipepline
from src.SubjectClassifier.pipeline.stage_03_training import Trainingpipeline
from src.SubjectClassifier.pipeline.stage_04_evaluation import Evaluationpipeline
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.text = "inputtext"
        self.classifier = Predictionpipeline(self.text)

@app.route("/index", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/data_ingestion", methods=['GET','POST'])
@cross_origin()
def data_ingestionRoute():
    try:
        STAGE_NAME="Data ingestion"
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        
        obj=DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
        
        return jsonify({"message": f"Stage {STAGE_NAME} completed successfully"}), 200
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500
    
@app.route("/prepare_fintune_model", methods=['GET','POST'])
@cross_origin()
def preparefinetuneRoute():
    STAGE_NAME = "Prepare fintune model"
    try: 
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = Preparefintunemodelpipepline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        
        return jsonify({"message": f"Stage {STAGE_NAME} completed successfully"}), 200
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500





@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainingRoute():
    try:
        STAGE_NAME="Training Base Model"
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        
        obj=Trainingpipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
        
        return jsonify({"message": f"Stage {STAGE_NAME} completed successfully"}), 200
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=['GET','POST'])
@cross_origin()
def evaluationRoute():
    try:
        STAGE_NAME="Model Evaluation"
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        
        obj=Evaluationpipeline()
        result = obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
        
        return jsonify({"message": f"Stage {STAGE_NAME} completed successfully", "result": result}), 200
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    text = request.json.get("text","")
    if text:
        try:
            clApp.text=text
            result = clApp.classifier.predict_using_cassical()
            logger.info("Prediction successful")
            return jsonify(result)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Prediction failed'}), 500
    else:
        logger.warning("Prediction attempted with no text provided")
        return jsonify({'error': 'No text provided'}), 400

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #for AWS