package kbPredict;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import kbPredict.MapSort;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class Predictor {
	/*
	 Logger,models and class lists are loaded once.
	 */
	
	 private static Map<String, String> kb= new HashMap<String, String>();
	 private static Classifier fusion_Classifier = new FilteredClassifier();
	 private static Classifier vCenter_Classifier = new FilteredClassifier();
	 private static Classifier vSphere_Classifier = new FilteredClassifier();
	 private static Classifier Horizonv_Classifier = new FilteredClassifier();
	 private static Classifier m_Classifier= new FilteredClassifier();
	 
	 private static ArrayList<String> fusion_cl = new ArrayList<String>();
	 private static ObjectInputStream fusion_model;
	 
	 private static ArrayList<String> vCenter_cl = new ArrayList<String>();
	 private static ObjectInputStream vCenter_model;
	 
	 private static ArrayList<String> vSphere_cl = new ArrayList<String>();
	 private static ObjectInputStream vSphere_model;
	 
	 private static ArrayList<String> Horizonv_cl = new ArrayList<String>();
	 private static ObjectInputStream Horizonv_model;
	 
	 private final static String config_file="/tmp/config.properties";
	 private static Properties props=new Properties();
	 
	 private static Logger logger = Logger.getLogger("NBMRest");

	 static {
	     try {
	    	 logger.setLevel(Level.ALL);

	         Formatter formatter = new Formatter() {
	             @Override
	             public String format(LogRecord arg0) {
	                 StringBuilder b = new StringBuilder();
	                 b.append(new Date());
	                 b.append(" ");
	                 b.append(arg0.getSourceClassName());
	                 b.append(" ");
	                 b.append(arg0.getSourceMethodName());
	                 b.append(" ");
	                 b.append(arg0.getLevel());
	                 b.append(" ");
	                 b.append(arg0.getMessage());
	                 b.append(System.getProperty("line.separator"));
	                 return b.toString();
	             }

	         };

	         Handler fh = new FileHandler("/tmp/NBMRest_logger.log");
	        
	         fh.setFormatter(formatter);
	         logger.addHandler(fh);

	         Handler[] handlers = logger.getHandlers();
	         if (handlers[0] instanceof ConsoleHandler) {
	           logger.removeHandler(handlers[0]);
	         }

            LogManager lm = LogManager.getLogManager();
	         lm.addLogger(logger);
	         logger.info("Initialized logger within static block");
	     }
	     catch (Throwable e) {
	         e.printStackTrace();
	     }
	 }

	 
	 static {
	    
				try {
					 
			    	 kb.clear();
					 fusion_cl.clear();
					 vCenter_cl.clear();
					 vSphere_cl.clear();
					 Horizonv_cl.clear();
					
				      //Load the models
						
				     InputStream in = new FileInputStream(config_file);
				     props.load(in);
				     in.close();
					
				     fusion_model = new ObjectInputStream(new FileInputStream(props.getProperty("fusion_modellocation")));
			    	 fusion_Classifier = (Classifier) fusion_model.readObject();
					 fusion_model.close();
				
					 vCenter_model = new ObjectInputStream(new FileInputStream(props.getProperty("vCenter_modellocation")));
			    	 vCenter_Classifier = (Classifier) vCenter_model.readObject();
					 vCenter_model.close();
			
					 vSphere_model = new ObjectInputStream(new FileInputStream(props.getProperty("vSphere_modellocation")));
			    	 vSphere_Classifier = (Classifier) vSphere_model.readObject();
					 vSphere_model.close();
				
					 Horizonv_model = new ObjectInputStream(new FileInputStream(props.getProperty("Horizonv_modellocation")));
			    	 Horizonv_Classifier = (Classifier) Horizonv_model.readObject();
					 Horizonv_model.close();
				
				

				} catch (Throwable e) {
					// Write to log
					logger.severe("Error loading Classifiers: " + e.toString());
					e.printStackTrace();
					
				}
				
				
			    //Load the KB article titles
						try {
							Path file = Paths.get(props.getProperty("KBlist"));
							try (InputStream in = Files.newInputStream(file);
							    BufferedReader reader =
							      new BufferedReader(new InputStreamReader(in))) {
							    String line = null;
							    while ((line = reader.readLine()) != null) {
							    	String[] strtmpArray = line.split("%%",2);
							    	kb.put(strtmpArray[0], strtmpArray[1]); //key, value
							    	
							    }  
							}
							
							//Load the classes - must be in same order as they were when model was created
							Path fclassFile = Paths.get(props.getProperty("fusion_classes"));
							try (InputStream fin = Files.newInputStream(fclassFile);
							    BufferedReader fclassReader =
							      new BufferedReader(new InputStreamReader(fin))) {
							      String fline = null;
							    while ((fline = fclassReader.readLine()) != null) {
							        fline.trim();
							       fusion_cl.add(fline);
							    } 	
							}
							     
							  //Load the classes - must be in same order as they were when model was created
							Path vcclassFile = Paths.get(props.getProperty("vCenter_classes"));
								try (InputStream vcin = Files.newInputStream(vcclassFile);
								    BufferedReader classReader =
								      new BufferedReader(new InputStreamReader(vcin))) {
								      String vcline = null;
								    while ((vcline = classReader.readLine()) != null) {
								        vcline.trim();
								      
								        
								        vCenter_cl.add(vcline);
								    } 	   
					    	
					    }
								
								 //Load the classes - must be in same order as they were when model was created
								Path vsclassFile = Paths.get(props.getProperty("vSphere_classes"));
									try (InputStream vsin = Files.newInputStream(vsclassFile);
									    BufferedReader classReader =
									      new BufferedReader(new InputStreamReader(vsin))) {
									      String vsline = null;
									    while ((vsline = classReader.readLine()) != null) {
									        vsline.trim();
									      
									        
									        vSphere_cl.add(vsline);
									    } 	   
						    	
						    }	
									
									//Load the classes - must be in SAME ORDER as they were when model was created
									Path hvclassFile = Paths.get(props.getProperty("Horizonv_classes"));
										try (InputStream hvin = Files.newInputStream(hvclassFile);
										    BufferedReader classReader =
										      new BufferedReader(new InputStreamReader(hvin))) {
										      String hvline = null;
										    while ((hvline = classReader.readLine()) != null) {
										        hvline.trim();
										       
										        Horizonv_cl.add(hvline);
										    } 	   
							    	
							    }		
							}
						   catch (Exception e) {
					
								// write to log
							   logger.severe("Error loading classes: " + e.toString());
							   e.printStackTrace();
									
								}
	    	 
	  
	     }
							
	public Map<String, String>  predict(String thetext, String theproduct) {
		 Map<String, String> map = new HashMap<String, String>();
		 ArrayList<String> cl = null;
		
		try {
			
			
			switch (theproduct) {
			case "Fusion":
			m_Classifier=fusion_Classifier;
				cl=fusion_cl;
			    break;
			case "vCenter":
			 m_Classifier=vCenter_Classifier;
				  cl=vCenter_cl;
			      break;
			case "vSphere":
			m_Classifier=vSphere_Classifier;
				  cl=vSphere_cl;
			      break;     
			case "Horizon View":
				 m_Classifier=Horizonv_Classifier;
				  cl=Horizonv_cl;
			      break;     	      
			      
			}
			
			
			FastVector fvNominalVal = new FastVector(cl.size());	
			
			for(int i=0;i< cl.size(); i++)	
		       {
		    	   fvNominalVal.addElement(cl.get(i));
		    	
		       }
		       
				
					Attribute attribute1 = new Attribute("Text", (FastVector) null); //text string
					Attribute attribute2 = new Attribute("KBID", fvNominalVal);
					
					// Create list of instances with one element
					FastVector fvWekaAttributes = new FastVector(2);
					fvWekaAttributes.addElement(attribute1);
					fvWekaAttributes.addElement(attribute2);
					
				        Instances test = new Instances("Test relation", fvWekaAttributes, 3);    //was 3 ?? 
					
				    
				
				       Instance instance = new Instance(2);
				
					
					instance.setValue(attribute1, thetext);
					// Another way to do it:
					// instance.setValue((Attribute)fvWekaAttributes.elementAt(0), text);
					test.add(instance);
				        test.setClassIndex(1);
					
				
			
				   double[] preddist=null;
				   double pred= m_Classifier.classifyInstance(test.instance(i));
			           test.classAttribute().value((int) pred);
					
					
              	preddist=m_Classifier.distributionForInstance(test.instance(i));
              	String predval=test.classAttribute().value((int) pred);
              	test.classAttribute().value((int) test.instance(i).classValue());
              	DecimalFormat df = new DecimalFormat("#.##########");
              	logger.info("product: " + theproduct + "| query: " + thetext +"| prediction: " + predval);
              	  
              		
              		
              	
              		 for (int j=0; j< test.instance(i).numClasses(); j++)
    				   {
              			String keyval=test.classAttribute().value(j); 
                                String thekey=keyval + " (" + kb.get(keyval)  +")"; //get KB ID (class) and the KB title
              			map.put(thekey, df.format(preddist[j])); //store the prediction value for this KB ID
    				   
    				   }
				 
	} catch (Exception e) {
		logger.severe("Error in predict: " + e.toString());
	        e.printStackTrace();
	    }
		
		 return MapSort.sortByValue(map);
		 

	} 

		
		
	
	}
	
		
		
	
	
	


