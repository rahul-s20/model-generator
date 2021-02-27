from dotenv import load_dotenv
from controllers import preprocessor, model
import os

load_dotenv(verbose=True)

if __name__ == "__main__":
    preprocess = preprocessor.Preprocessing(aws_secret=os.environ['secret'], aws_access=os.environ['access'],
                                            end_point=os.environ['OVERRIDE_S3_ENDPOINT'], region_name=os.environ['region'])
    x,y = preprocess.process_data_from_s3(src_bucket=os.environ['SOURCE_BUCKET'], categories=os.environ['CATEGORIES'])
    trainX, testX, trainY, testY = preprocess.one_hot_encoding_train_test
    modl, aug = model.prepare_model()
    model.train_and_save_model(modl, aug, trainX, trainY, testX, testY)

