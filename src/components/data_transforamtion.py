import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DatatransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DatatransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ["writing_score","reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logging.info("Numerical columns Scaling is completed")
            logging.info("Categorical columns encoding is completed")

            preprocessor =ColumnTransformer(
                [
                    ("num_features",num_pipeline,numerical_features),
                    ("cat_features",cat_pipeline,categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df= pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("read train and test data completed")
            logging.info("Obtaining preprocessing object")

            processing_obj =self.get_data_transformer_object()
            target_columns ="math_score"
            numerical_features = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_columns],axis=1)
            target_feature_train_df = train_df[target_columns]

            input_feature_test_df = test_df.drop(columns=[target_columns],axis=1)
            target_feature_test_df = test_df[target_columns]

            logging.info(f"Applying preprocesssing object  on training dataframe and testing dataframe. ")

            input_feature_train_arr =processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr =processing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved Preprocessing object")
            
            save_object (
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=processing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

