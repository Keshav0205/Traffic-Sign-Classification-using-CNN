from traffic_sign_cnn import traffic_sign_classifier
import matplotlib.pyplot as plt
tfsc = traffic_sign_classifier()
import numpy as np
tfsc.print_shape()

# tfsc.plot_grid(5,5)
#tfsc.gray_normalization()

#tfsc.build_CNN_model(dropout=0.4)
#accuracy = 
#tfsc.CNN.summary()
#tfsc.load_model_data(load_path = 'model_dropout_4.h5')
#tfsc.compile_and_train()
#tfsc.test_plot()
#tfsc.save_model(save_path = 'model_dropout_4.h5')
#tfsc.plot_metrics()
#tfsc.evaluate_model()
#tfsc.plot_metrics()