import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayes;

import java.io.FileReader;
import java.io.BufferedReader;

public class train_save_weka_MLP_model{

     public static void main (String[] args) throws Exception{

          /* sets the file to use for training */
		     BufferedReader m_Training = new BufferedReader (new FileReader ("C:/UNO_Courses/Weka_API_coding/iris_training.arff"));
             Instances Train = new Instances (m_Training);
             Train.setClassIndex(Train.numAttributes() - 1); // To indicate the output column (Y) in the training Table which is the last column

             m_Training.close(); // This is not needed, so better close it

             MultilayerPerceptron mlp = new MultilayerPerceptron();   //Create an MultilayerPerceptron Object

             mlp.setOptions(weka.core.Utils.splitOptions("-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a"));
             mlp.buildClassifier(Train);          //Train the model

             weka.core.SerializationHelper.write("saved_mlp_model",mlp); // Save the model in a file to be loaded and used later

             System.out.println("Trained model is created and saved successfully!");

		 }

	}
