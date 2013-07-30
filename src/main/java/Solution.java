import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.BayesianLogisticRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.NBTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.*;


/**
 * This class describes a topic in the Quora data set.
 * @author sholtzen
 *
 */
class Topic {
	public String text;
	public int followers;
	public int numanswers;
	public int numtrue;
	
	public Topic(String Text, int Followers) {
		this.text = Text;
		this.followers = Followers;
		numanswers = 0;
		numtrue = 0;
	}
	
	/**
	 * Called to update an answer count. This is used to determine the answer percent rate.
	 * @param answer
	 */
	public void addAnswer(Boolean answer) {
		if(answer) {
			numtrue++;
		}
		numanswers++;
	}
	
	/**
	 * @return the percentage of how often questions that are a member of this topic get answered.
	 */
	public double getPercentAnswered() {
		double res = (double)numtrue / numanswers;
        if (res != Float.NaN)
            return res;
        else
            return 0;
	}
	
	/**
	 * A static variable that holds the list of topics. 
	 */
	public static ArrayList<Topic> topicList = new ArrayList<Topic>();
	private static int currentId = 0;
	/**
	 * Call this method in order to ensure that the topic is in the topic list.
	 * @param text
	 * @param Followers
	 * @return
	 */
	public static Topic getTopic(String text, int Followers){
		for(Topic t : topicList) {
			if (t.text.equals(text))
				return t;
		}
		topicList.add(new Topic(text, Followers));
		return topicList.get(topicList.size() - 1);
	}
	
	public String toString() {
		return "";
	}
}

/**
 * This class represents Quora question submissions.
 * @author sholtzen
 *
 */
class QuoraSubmission {
	public String text;
	public Topic context_topic;
	public ArrayList<Topic> topic_list;
	public boolean anonymous;
	public String question_key;
	public Boolean answer_class;
	
	public QuoraSubmission(String text,Topic context_topic,
			ArrayList<Topic> topic_list, boolean anonymous,
			String question_key, Boolean answer_class) {
		super();
		this.text = text.replaceAll("\"", "").replaceAll("\n", "");
		this.topic_list = topic_list;
		this.anonymous = anonymous;
		this.question_key = question_key;
		this.answer_class = answer_class;
		this.context_topic = context_topic;
	}
	
	// Returns the submission in ARFF comma seperated format
	public String toString() {
		String res = "\"" + this.text + "\",";
		double answerprob = 0;
		for( Topic t : topic_list) {
			answerprob += t.getPercentAnswered();
		}
		res += answerprob+",";
		int total_followers = 0;
		int max = 0;
		for (Topic t : topic_list) {
			total_followers += t.followers;
			if(t.followers > max)
				max = t.followers;
		}
		res += ""+total_followers+",";
		res += ""+anonymous+",";
		res += ""+topic_list.size()+",";
		res += ""+max+",";
		if(context_topic != null)
			res += ""+context_topic.followers+",";
		else
			res += "0,";
		res += text.length()+",";
		res += ""+answer_class;
		return res;
	}
	
	/**
	 * Returns an instance, which is a Weka formatted object that it uses in order to form a classifaction.
	 * @param attributes, a formatted fast vector that contains all of the attributes for the instance
	 * @return
	 */
	public Instance getInstance(FastVector attributes) {
		Instance i = new Instance(9);
		double answerprob = 0;
		int total_followers = 0;
		int max = 0;
        
		for( Topic t : topic_list) {
            if(t.getPercentAnswered() > answerprob) 
                answerprob = t.getPercentAnswered();
			total_followers += t.followers;
			if(t.followers > max)
				max = t.followers;
		}
		i.setValue((Attribute)attributes.elementAt(0), answerprob);
		i.setValue((Attribute)attributes.elementAt(1), total_followers);
		i.setValue((Attribute)attributes.elementAt(4), max);
		i.setValue((Attribute)attributes.elementAt(2), String.valueOf(anonymous));
		i.setValue(3, this.topic_list.size());
		if(context_topic != null)
			i.setValue((Attribute)attributes.elementAt(5), context_topic.followers);
		else
			i.setValue((Attribute)attributes.elementAt(5), 0);
		i.setValue((Attribute)attributes.elementAt(6), text.length());
		if(this.answer_class != null) 
			i.setValue((Attribute)attributes.elementAt(7), String.valueOf(this.answer_class));
			
		i.setValue((Attribute)attributes.elementAt(8), String.valueOf(this.context_topic != null));
		return i;
	}
}

public class Solution 
{
    public static void main( String[] args ) throws Exception
    {
    	// Create two lists, one for training and one for testing data.
    	ArrayList<QuoraSubmission> training_submissions = new ArrayList<QuoraSubmission>();
    	ArrayList<QuoraSubmission> test_submissions = new ArrayList<QuoraSubmission>();
    	
        BufferedReader stdin = null;
		stdin = new BufferedReader(new InputStreamReader(System.in));
        String line;
        int numsamples = 0;
		line = stdin.readLine();
		numsamples = Integer.valueOf(line);
        JSONParser parser = new JSONParser();
        
        
        // first build the test set
        for(int i = 0; i < numsamples; i++) {
        	line = stdin.readLine();
        	JSONObject result = (JSONObject)parser.parse(line);
        	JSONObject context_topic = (JSONObject)result.get("context_topic");
        	ArrayList<Topic> topicList = new ArrayList<Topic>();
        	
        	// get the topics
        	for (Object obj : (JSONArray)result.get("topics")) {
        		JSONObject cur = (JSONObject)obj;
        		Topic t = Topic.getTopic(((String)cur.get("name")), Integer.valueOf(cur.get("followers").toString()));
        		t.addAnswer((Boolean)result.get("__ans__"));
        		topicList.add(t);
        	}
        	if(context_topic != null)
        		training_submissions.add(new QuoraSubmission((String)result.get("question_text"), 
        			new Topic((String)context_topic.get("name"), Integer.valueOf(context_topic.get("followers").toString())),
        			topicList, (Boolean)result.get("anonymous"), (String)result.get("question_key"), (Boolean)result.get("__ans__")));
        	else
        		training_submissions.add(new QuoraSubmission((String)result.get("question_text"), null,
            			topicList, (Boolean)result.get("anonymous"), (String)result.get("question_key"), (Boolean)result.get("__ans__")));
        }
        
        // now get the training set
        int numtrainingset = Integer.valueOf(stdin.readLine());
        for(int i = 0; i < numtrainingset; i++) {
        	line = stdin.readLine();
        	JSONObject result = (JSONObject)parser.parse(line);
        	JSONObject context_topic = (JSONObject)result.get("context_topic");
        	ArrayList<Topic> topicList = new ArrayList<Topic>();
        	
        	// get the topics
        	for (Object obj : (JSONArray)result.get("topics")) {
        		JSONObject cur = (JSONObject)obj;
        		Topic t = Topic.getTopic(((String)cur.get("name")), Integer.valueOf(cur.get("followers").toString()));
        		topicList.add(t);
        	}
        	if(context_topic != null)
        		test_submissions.add(new QuoraSubmission((String)result.get("question_text"), 
        			new Topic((String)context_topic.get("name"), Integer.valueOf(context_topic.get("followers").toString())),
        			topicList, (Boolean)result.get("anonymous"), (String)result.get("question_key"), null));
        	else
        		test_submissions.add(new QuoraSubmission((String)result.get("question_text"), null,
            			topicList, (Boolean)result.get("anonymous"), (String)result.get("question_key"), null));
        }
        
        
        // set up attributes
        Attribute topicprob = new Attribute("topic_probability");	// the magic one!
        Attribute totalfollows = new Attribute("total_follows");
        Attribute numtopics = new Attribute("num_topics");
        Attribute mostpopulartopic = new Attribute("most_popular_topic");
        Attribute contexttopicfollowers = new Attribute("context_topic_followers");
        Attribute questionlength = new Attribute("question_length");
        
        FastVector fv = new FastVector(2);
        fv.addElement("false");
        fv.addElement("true");
        Attribute hascontextclass = new Attribute("hasscontext", fv);
        Attribute answer = new Attribute("class", fv);
        Attribute anonymous = new Attribute("anonymous", fv);

        FastVector features = new FastVector(9);
        features.addElement(topicprob); 
        features.addElement(totalfollows);
        features.addElement(anonymous);
        features.addElement(numtopics);
        features.addElement(mostpopulartopic);
        features.addElement(contexttopicfollowers);
        features.addElement(questionlength);
        features.addElement(answer);
        features.addElement(hascontextclass);
        
        // set up filter
        Remove remove = new Remove();
        remove.setAttributeIndices("1");
        
        Instances trainingset = new Instances("QuoraQuestionAnswers", features, 9);
        Discretize d = new Discretize();
        d.setInputFormat(trainingset);
        trainingset.setClassIndex(7);
        trainingset = Filter.useFilter(trainingset, d);
        
        for (QuoraSubmission q : training_submissions) {
        	trainingset.add(q.getInstance(features));
        }
        Classifier tree_with_topics = (Classifier)new SMO();
        FilteredClassifier tree_without_topics = new FilteredClassifier();
        tree_without_topics.setClassifier(new SMO());
        Remove r = new Remove();
        r.setAttributeIndices("1");
        tree_without_topics.setFilter(r);
        try {
			tree_with_topics.buildClassifier(trainingset);
			tree_without_topics.buildClassifier(trainingset);
	        Evaluation eTest = new Evaluation(trainingset);
	        eTest.evaluateModel(tree_with_topics, trainingset);
	        String strSummary = eTest.toSummaryString();
	        System.out.println(strSummary);
            
 	        eTest = new Evaluation(trainingset);
	        eTest.evaluateModel(tree_without_topics, trainingset);
	        strSummary = eTest.toSummaryString();
	        System.out.println(strSummary);
            
			for (QuoraSubmission q: test_submissions) {
				Classifier c = null;
				Instance i = q.getInstance(features);
                if(i.value(topicprob) != 0) {
                    //System.out.println("Using topic tree");
					c = tree_with_topics;
                }
                else {
                    //System.out.println("Using non-topic tree");
					c = tree_without_topics;
                }
                i.setDataset(trainingset);
				String out;
				if(c.classifyInstance(i) == 0)
					out = "false";
				else
					out = "true";
				System.out.println("{\"__ans__\":" + out + ",\"question_key\":\"" +
						q.question_key +"\"}");
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
    }
}
