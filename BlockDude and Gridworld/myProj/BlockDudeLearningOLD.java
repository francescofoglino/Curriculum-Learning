package myProj;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.valuefunction.QValue;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.domain.singleagent.blockdude.state.BlockDudeAgent;
import burlap.domain.singleagent.blockdude.state.BlockDudeCell;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.domain.singleagent.blockdude.state.BlockDudeMap;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

public class BlockDudeLearning {
	
	//block dude domain
	BlockDude bd;
	OOSADomain domain;
	RewardFunction rf;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	
	//constructor
	public BlockDudeLearning(){
		
		bd = new BlockDude(10,10);
		//specify map here
		tf = new BlockDudeTF();
		bd.setTf(tf);
		goalCondition = new TFGoalCondition(tf);
		domain = bd.generateDomain();
		
		//setup initial state
		
		//List<BlockDudeCell> obstacles = new ArrayList();
		//obstacles.add(new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block"));
		//obstacles.add(new BlockDudeCell(6, 1, BlockDude.CLASS_BLOCK, "block"));
		
		//REMEMBER: the map has to forbid the agent to put a block outside the world
		initialState = new BlockDudeState(new BlockDudeAgent(1, 1, 0, false), 
										new BlockDudeMap(new int[][] {{1,1,1,1,1,1,0,0,0,0},
																	  {1,0,0,0,0,0,0,0,0,0},
																	  {1,0,0,0,0,0,0,0,0,0},
																	  {1,0,0,0,0,0,0,0,0,0},
																	  {1,0,0,0,0,0,0,0,0,0},
																	  {1,0,0,0,0,0,0,0,0,0},
																	  {1,0,0,0,0,0,0,0,0,0},
																	  {1,0,0,0,0,0,0,0,0,0},
																	  {1,1,1,1,0,0,0,0,0,0},
																	  {1,0,0,0,0,0,0,0,0,0}}), 
										new BlockDudeCell(9, 1, BlockDude.CLASS_EXIT, "exit"),
										new BlockDudeCell(5, 1, BlockDude.CLASS_BLOCK, "block"));
		
		//instantiate the HashableStateFactory
		hashingFactory = new SimpleHashableStateFactory();
		
		//creation of the actual environment with which our agent will interact
		env = new SimulatedEnvironment(domain, initialState);
	}
	
	public void visualize(String outputPath){
		Visualizer v = BlockDudeVisualizer.getVisualizer(10, 10);
		new EpisodeSequenceVisualizer(v, domain, outputPath);
	}
	
	//function to merge in one the various examples available on the website http://burlap.cs.brown.edu/tutorials/bpl/p4.html#sarsa
	public void PerformLearningAlgorithm(String outputPath, int algorithmChoice){
		
		//clean directory
		File currentFolder = new File(outputPath);
		currentFolder.delete();
		
		QLearning agent = null;
		String writeString = "";
		
		//selection and initialization of the learning agent
		if(algorithmChoice == 0){
			System.out.println("You have chosen Q-Learning");
			agent = new QLearning(domain, 0.999, hashingFactory, 0., 1.);
			writeString = "ql_";
		}else if(algorithmChoice == 1){
			System.out.println("You have chosen Sarsa(lambda)");
			agent = new SarsaLam(domain, 0.999, hashingFactory, 0., 0.2, 0.9);
			writeString = "sarsa_";
		}else{
			System.out.println("Your algorithm choice (" + writeString + ") is not available. "
					+ "Please select one of the followings:"
					+ "\n - 0 for Q-Learning"
					+ "\n - 1 for Sarsa(lambda)");
			return;
		}
		
		//writer for Action Values
		PrintWriter osAV = null;
		try {
			osAV = new PrintWriter(outputPath+"/ActionValueFunction.txt");
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		//writer for Return
		PrintWriter osR = null;
		try {
			osR = new PrintWriter(outputPath+"/Return.txt");
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		//some initializations for limiting the overhead
		Episode e = null; //object for storing the episode at each loop
		List<Double> oneLineToPrint = new ArrayList();//List for saving all the Action Values relative to S0
		List<QValue> initQValues = new ArrayList();//List for storing the Q-values in S0
		
		//iterator for the loop for collecting the Action Values
		ListIterator<QValue> itr = null;
		
		//run learning for 40.000 episodes
		for(long i = 0; i < 5; i++){
			
			/*
			if(i==450){
				agent.setLearningPolicy(new GreedyDeterministicQPolicy(agent));
			}
			*/
			System.out.println("dio");
			//the 100 is the maximum number of steps to avoid the episode to run infinitely
			e = agent.runLearningEpisode(env,100);
			e.write(outputPath + writeString + i);
			
			//the first element is the number of the episode
			oneLineToPrint.add(new Double(i));
			
			initQValues = agent.qValues(initialState);
			
			System.out.println("before for");
			
			for(itr = initQValues.listIterator(); itr.hasNext() ;){
				QValue currentQValue = itr.next();
				//String actionName = currentQValue.a.actionName();
				double actionValue = currentQValue.q;
				//String cQVString = new String(currentQValue.s.toString().substring(50, 93).replace("/n", " "));
				//System.out.println("State : " + "State 0" + " Action : " + actionName + " value : " + actionValue);
				System.out.println(actionValue);
				oneLineToPrint.add(actionValue);
			}
			//build the string for printing out a file
			String oneLineToPrintString = oneLineToPrint.toString().replace("],", "\n").replace("[", "").replace("]", "").replace(" ", "").replace(",", " ");	
			osAV.println(oneLineToPrintString);		
			
			osR.println(e.discountedReturn(0.999));
			
			agent.writeQTable(outputPath+"/TIME_END.txt");
			
			//reset environment for next learning episode
			env.resetEnvironment();
			
			oneLineToPrint.clear();
			initQValues.clear();
		}
		
		osAV.close();
		osR.close();
		
	}
	
	/*public void manualValueFunctionVis(ValueFunction valueFunction*//**, Policy p*//*){
		
		List<State> allStates = StateReachability.getReachableStates(
				initialState, domain, hashingFactory);

		//define color function
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		//define a 2D painter of state values, specifying 
		//which variables correspond to the x and y coordinates of the canvas
		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYKeys("agent:x", "agent:y",	new VariableDomain(0, 10), new VariableDomain(0, 10), 1, 1);

		//create our ValueFunctionVisualizer that paints for all states
		//using the ValueFunction source and the state value painter we defined
		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, valueFunction);
		
		//set the background color for places where states are not rendered to grey
		gui.setBgColor(Color.GRAY);

		//start it
		gui.initGUI();


	}*/
	
	/*public void valueIterationExample(String outputPath){

		Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 100);
		//Policy p = planner.planFromState(initialState);

		//PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");

		manualValueFunctionVis((ValueFunction)planner*//**, p*//*);

	}*/
	
	public static void main(String[] args) {
		System.out.println("Welcome to BlockDude Learning!");
		
		BlockDudeLearning example = new BlockDudeLearning();
		String outputPath = "output/"; //directory to record results
		
		System.out.println("Starting Learning phase...");
		
		//block for the creation of an empty file in order to
		//keep track of the actual time for the learning start
		PrintWriter timeFile = null;
		try {
			timeFile = new PrintWriter(outputPath+"/TIME_BEGIN.txt");
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		timeFile.close();
		
		//learning algorithms hereqValues
		example.PerformLearningAlgorithm(outputPath,1);
		
		//for visualization
		//example.valueIterationExample(outputPath);
		
		//run the visualizer
		example.visualize(outputPath);
		
		return;
	
	}
	
	
}
