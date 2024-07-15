
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;

import java.io.FileReader;
import java.io.BufferedReader;

public class load_saved_weka_MLP_model_to_predict{

     public static void main (String[] args) throws Exception{


              MultilayerPerceptron mlp = new MultilayerPerceptron();  // Create a MultilayerPerceptron Object
              mlp = (MultilayerPerceptron)weka.core.SerializationHelper.read("saved_mlp_model"); // Read the save model from a file. Note, the type casting is needed here.

             BufferedReader m_Test = new BufferedReader (new FileReader ("C:/UNO_Courses/Weka_API_coding/iris_test1.arff")); //Load the Test file
             Instances Test = new Instances (m_Test);
             Test.setClassIndex(Test.numAttributes() - 1); // To indicate the output column (Y='?') in the Test Table, which is the last column
             m_Test.close(); // Not needed, so better close it

            for (int i=0;i<Test.size();i++){
             weka.core.Instance LoadOneTest = Test.instance(i);
             System.out.println(LoadOneTest);
             double TestOutput = mlp.classifyInstance(LoadOneTest);
             System.out.println("The predicted class is, "+ (int)TestOutput);
		    }


		 }

	}
