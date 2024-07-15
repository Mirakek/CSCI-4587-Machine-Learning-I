import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayes;

import java.io.FileReader;
import java.io.BufferedReader;

public class load_saved_weka_NB_model_to_predict{

     public static void main (String[] args) throws Exception{


              NaiveBayes nb = new NaiveBayes();  // Create a NaiveBayes Object
              nb = (NaiveBayes)weka.core.SerializationHelper.read("saved_nb_model"); // Read the save model from a file. Note, the type casting is needed here.

             BufferedReader m_Test = new BufferedReader (new FileReader ("C:/UNO_Courses/Weka_API_coding/iris_test1.arff")); //Load the Test file
             Instances Test = new Instances (m_Test);
             Test.setClassIndex(Test.numAttributes() - 1); // To indicate the output column (Y='?') in the Test Table, which is the last column
             m_Test.close(); // Not needed, so better close it

            for (int i=0;i<Test.size();i++){
             weka.core.Instance LoadOneTest = Test.instance(i);
             System.out.println(LoadOneTest);
             double TestOutput = nb.classifyInstance(LoadOneTest);
             System.out.println("The predicted class is, "+ (int)TestOutput);
		    }


		 }

	}
