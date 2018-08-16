# -*- coding: utf-8 -*-
"""
BaseModel defines basic functions for various classifier.
"""
import abc
import os
import io
import logging
import pickle


class BaseModel(abc.ABC):
    """
    Wrapper of BaseModel that will be inherited.
    """
    def __init__(self):
        self.model = None

    def predict(self, raw_x):
        """
        Return a list of predicted classification.
        """
        test_x = self.transform_raw_x(raw_x)
        return self.model.predict(test_x)

    def predict_proba(self, raw_x):
        """
        Return a list of predicted probability for each classification.
        """
        test_x = self.transform_raw_x(raw_x)
        return self.model.predict_proba(test_x)

    def train(self, raw_x, raw_y):
        """
        Train self.model.
        """
        logging.info("Transform test data to fit our model.")
        train_x = self.transform_raw_x(raw_x)

        logging.info(train_x.shape)
        train_y = self.transform_raw_y(raw_y)
        logging.info(train_y.shape)
        self.model = self._init_model()

        logging.info("Train model")
        self.model.fit(train_x, train_y)

    @abc.abstractmethod
    def _init_model(self):
        pass

    @abc.abstractmethod
    def transform_raw_x(self, raw_x):
        """
        Transform |raw_x| into the format that fits to a classifier.
        """
        pass

    @abc.abstractmethod
    def transform_raw_y(self, raw_y):
        """
        Transform |raw_y| into the format that fits to a classifier.
        """
        pass

    def save(self, fpath):
        """
        Save model to the file |fpath|.
        """
        logging.warning("Save model to %s", fpath)
        _dir = os.path.dirname(fpath)
        if not os.path.isdir(_dir):
            os.makedirs(_dir)

        with io.open(fpath, "wb") as _fp:
            pickle.dump(self, _fp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fpath):
        """
        Load model to from the file |fpath|.
        """
        ret = None
        if os.path.isfile(fpath):
            logging.warning("Load from %s", fpath)
            with io.open(fpath, "rb") as _fp:
                ret = pickle.load(_fp)
        else:
            logging.warning("Cannot load model from %s", fpath)

        return ret
