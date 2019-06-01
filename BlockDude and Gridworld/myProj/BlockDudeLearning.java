package myProj;

import java.awt.Color;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import burlap.behavior.functionapproximation.DifferentiableStateActionValue;
import burlap.behavior.functionapproximation.dense.ConcatenatedObjectFeatures;
import burlap.behavior.functionapproximation.dense.NumericVariableFeatures;
import burlap.behavior.functionapproximation.sparse.LinearVFA;
import burlap.behavior.functionapproximation.sparse.SparseStateFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TileCodingFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.behavior.learningrate.ConstantLR;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.GreedyDeterministicQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.behavior.valuefunction.ValueFunction;
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
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.common.UniformCostRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
//import learning.transfer.ExtendedShapingRewardFunction;
//import learning.transfer.advice.NextAction;
//import learning.transfer.tdmethods.GradientDescentSarsaLamTransfer;

public class BlockDudeLearning{
	
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
											  {1,0,0,0,0,0,0,0,0,0},
											  {1,1,1,1,1,1,0,0,0,0}}), 
				new BlockDudeCell(8, 1, BlockDude.CLASS_EXIT, "exit"));
		
		//instantiate the HashableStateFactory
		hashingFactory = new SimpleHashableStateFactory();
		
		//creation of the actual environment with which our agent will interact
		env = new SimulatedEnvironment(domain, initialState);
	}
	
	public void visualize(String outputPath){
		Visualizer v = BlockDudeVisualizer.getVisualizer(10, 10);
		new EpisodeSequenceVisualizer(v, domain, outputPath);
	}
	
	//function to merge in one the various examples available on the web site http://burlap.cs.brown.edu/tutorials/bpl/p4.html#sarsa
	public void PerformLearningAlgorithm(String outputPath, int algorithmChoice){
		OurLinearVFA vfa = null;
		
		//clean directory
		File currentFolder = new File(outputPath);
		currentFolder.delete();
		
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
		
		//writer for Return
		PrintWriter osNLR = null;
		try {
			osNLR = new PrintWriter(outputPath+"/NoLearningReturn.txt");
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		//writer for VFA, it is useful just if algorithm choice is equal to 3
		PrintWriter osVFA = null;
		try {
			osVFA = new PrintWriter(outputPath+"/Sources.txt");
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		for(int epoch=0; epoch<10; ++epoch){	
			
			LearningAgent agent = null;
			String writeString = "";
			
			//selection and initialisation of the learning agent
			if(algorithmChoice == 0){
				System.out.println("You have chosen Q-Learning");
				agent = new QLearning(domain, 0.999, hashingFactory, 0., 1.);
				writeString = "ql_";
			}else if(algorithmChoice == 1){
				System.out.println("You have chosen Sarsa(lambda)");
				agent = new SarsaLam(domain, 0.999, hashingFactory, 0., 0.1, 0.9);
				writeString = "sarsa_";
			}else if(algorithmChoice == 2){
				System.out.println("You have chosen Sarsa(lambda) VA");
				
				//construction of the list for specifying the variables on which we want to build our tiles
				//REMEMBER the number of variables depends on the world you are implementing
				List<Object> stateVariablesList = new ArrayList<Object>();
				stateVariablesList.add("agent:x");stateVariablesList.add("agent:y");
				stateVariablesList.add("agent:dir");stateVariablesList.add("agent:holding");
				stateVariablesList.add("block1:x");stateVariablesList.add("block1:y");
				stateVariablesList.add("block2:x");stateVariablesList.add("block2:y");
				stateVariablesList.add("block3:x");stateVariablesList.add("block3:y");
				
				NumericVariableAndBooleanFeatures stateVariables = new NumericVariableAndBooleanFeatures(stateVariablesList);
				
				TileCodingFeatures tileCoding = new TileCodingFeatures(stateVariables);
				
				double[] tilingWidths = new double[stateVariablesList.size()];
				tilingWidths[0] = 3; tilingWidths[1] = 3; 
				tilingWidths[2] = 0.5;tilingWidths[3] = 0.5; 
				tilingWidths[4] = 3; tilingWidths[5] = 3;
				tilingWidths[6] = 3; tilingWidths[7] = 3;
				tilingWidths[8] = 3; tilingWidths[9] = 3;
				
				tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, 8, TilingArrangement.RANDOM_JITTER);
				
				agent = new GradientDescentSarsaLam(domain, 0.999, new LinearVFA(tileCoding), 0.1/(tileCoding.features(initialState).size()), 0.9);
				writeString = "stl_";
			}else if(algorithmChoice == 3){
				System.out.println("You have chosen Sarsa(lambda) VA personally modified");
				
				//construction of the list for specifying the variables on which we want to build our tiles
				Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
				domains.put("block1:x", new VariableDomain(-7,4));domains.put("block1:y", new VariableDomain(-3,1));
				domains.put("block2:x", new VariableDomain(-7,4));domains.put("block2:y", new VariableDomain(-3,1));
				domains.put("block3:x", new VariableDomain(-7,4));domains.put("block3:y", new VariableDomain(-3,1));
				domains.put("exit:x", new VariableDomain(0,7));domains.put("exit:y", new VariableDomain(-3,0));
				domains.put("agent:dir",new VariableDomain(0,1));domains.put("agent:holding",new VariableDomain(0,1));
				
				NormalizedVariableFeaturesBlockDude stateVariables = new NormalizedVariableFeaturesBlockDude(domains);
				
				TileCodingFeatures tileCoding = new TileCodingFeatures(stateVariables);
				
				double[] tilingWidths = new double[domains.size()];
				tilingWidths[0] = 3/11; tilingWidths[1] = 3/4; 
				tilingWidths[2] = 3/11;tilingWidths[3] = 3/4; 
				tilingWidths[4] = 3/11; tilingWidths[5] = 3/4;
				tilingWidths[6] = 3/7; tilingWidths[7] = 1;//tilingWidths[0] = 3/7; tilingWidths[1] = 1;
				tilingWidths[8] = 0.5; tilingWidths[9] = 0.5;//tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;
				
				tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, 8, TilingArrangement.RANDOM_JITTER);
				
				vfa = new OurLinearVFA(tileCoding);//we save it here to use it for transfer
				
				agent = new GradientDescentSarsaLam(domain, 0.999, vfa, 0.1/(tileCoding.features(initialState).size()), 0.9);
				writeString = "stlN_";
			}else{
				System.out.println("Your algorithm choice (" + writeString + ") is not available. "
						+ "Please select one of the followings:"
						+ "\n - 0 for Q-Learning"
						+ "\n - 1 for Sarsa(lambda)");
				return;
			}
			
			QProvider qp = (QProvider)agent;
						
			//run learning for 40.000 episodes
			for(long i = 0; i < 200; i++){
				
				//write action values
				osAV.print(i+ " ");
				List<QValue> initQValues = qp.qValues(initialState); //S0
				for(ListIterator<QValue> itr = initQValues.listIterator(); itr.hasNext() ;){
					QValue currentQValue = itr.next();
					double actionValue = currentQValue.q;
					
					osAV.print(actionValue+ " ");
				}
				osAV.println();
				
				/*
				if(i%10 == 0){
					PerformCurrentPolicy(i, agent, osNLR);
					
					agent.setLearningPolicy(new EpsilonGreedy(agent, 0.1));
					agent.setLearningRateFunction(new ConstantLR(.1));
				}
				*/
				
				//kill exploration after a specified number of episodes
//				if(i>=200){
//					if(agent instanceof QLearning){
//						double param = 0.1 * (600 - i)/400;
//						QLearning agentQLearning = (QLearning)agent;
//						agentQLearning.setLearningPolicy(new EpsilonGreedy(agentQLearning, param));
//					}else if(agent instanceof GradientDescentSarsaLam){
//						double param = 0.1 * (600 - i)/400;
//						GradientDescentSarsaLam agentQLearning = (GradientDescentSarsaLam)agent;
//						agentQLearning.setLearningPolicy(new EpsilonGreedy(agentQLearning, param));
//					}else{
//						System.out.println("ERROR! You are trying to change the learning policy to an invalid agent!");
//					}
//				}
				
				Episode e = agent.runLearningEpisode(env,50);//object for storing the episode at each loop
				
				e.write(outputPath + writeString + "_" + epoch + "_" + i);			
				
				osR.println(i + " " + e.discountedReturn(1));
				
				//agent.writeQTable(outputPath+"/TIME_END.txt");mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
				
				//reset environment for next learning episodeQLearning a = 
				env.resetEnvironment();
			}
			if(algorithmChoice==3){
				ArrayList<Double> paramToSave = new ArrayList<Double>();
				System.out.println(vfa.numParameters());
				for(int p = 0; p < vfa.numParameters(); ++p){
					paramToSave.add((vfa.getParameter(p)));
				}
				osVFA.println(paramToSave.toString());
				
				vfa.getKeys().stream().forEach(System.out::println);
				System.out.println(vfa.getKeys().size());
			}
			
		}	
		
		osAV.close();
		osR.close();
		osNLR.close();
		osVFA.close();
		
		//return vfa;
		
	}


//	//method for TL algorithms
//	public void PerformTransferLearningAlgorithm(String outputPath, String sourcesPath, LinearVFA source){
//		/************************  IMPORTANT PARAMETRISATION  *************************/
//		double gamma = 1.;
//		double alpha = 0.;//will be updated
//		double lambda = 0.9;
//		double epsilon = 0.1;
//		/************************************  END  ***********************************/
//		
//		//clean directory
//		File currentFolder = new File(outputPath);
//		currentFolder.delete();
//		
//		//writer for Action Values
//		PrintWriter osAV = null;
//		try {
//			osAV = new PrintWriter(outputPath+"/ActionValueFunction.txt");
//		} catch (FileNotFoundException e1) {
//			// TODO Auto-generated catch block
//			e1.printStackTrace();
//		}
//		
//		//writer for Return
//		PrintWriter osR = null;
//		try {
//			osR = new PrintWriter(outputPath+"/Return.txt");
//		} catch (FileNotFoundException e1) {
//			// TODO Auto-generated catch block
//			e1.printStackTrace();
//		}
//		
//		//writer for Return
//		PrintWriter osNLR = null;
//		try {
//			osNLR = new PrintWriter(outputPath+"/NoLearningReturn.txt");
//		} catch (FileNotFoundException e1) {
//			// TODO Auto-generated catch block
//			e1.printStackTrace();
//		}
//		
//		System.out.println("You have chosen Sarsa(lambda) VA personally modified");
//		
//		//construction of the list for specifying the variables on which we want to build our tiles
////		Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
//		//domains.put("block1:x", new VariableDomain(-7,4));domains.put("block1:y", new VariableDomain(-3,1));
//		//domains.put("block2:x", new VariableDomain(-7,4));domains.put("block2:y", new VariableDomain(-3,1));
//		//domains.put("block3:x", new VariableDomain(-7,4));domains.put("block3:y", new VariableDomain(-3,1));
////		domains.put("exit:x", new VariableDomain(0,7));domains.put("exit:y", new VariableDomain(-3,0));
////		domains.put("agent:dir",new VariableDomain(0,1));domains.put("agent:holding",new VariableDomain(0,1));
//		
////		NormalizedVariableFeaturesBlockDude stateVariables = new NormalizedVariableFeaturesBlockDude(domains);
//		
////		TileCodingFeatures tileCoding = new TileCodingFeatures(stateVariables);
//		
////		double[] tilingWidths = new double[domains.size()];
//		//tilingWidths[0] = 3/11; tilingWidths[1] = 3/4; 
//		//tilingWidths[2] = 3/11;tilingWidths[3] = 3/4; 
//		//tilingWidths[4] = 3/11; tilingWidths[5] = 3/4;
////		tilingWidths[0] = 3/7; tilingWidths[1] = 1;//tilingWidths[6] = 3/7; tilingWidths[7] = 1;
////		tilingWidths[2] = 0.5; tilingWidths[3] = 0.5;//tilingWidths[8] = 0.5; tilingWidths[9] = 0.5;
//		
////		tileCoding.addTilingsForAllDimensionsWithWidths(tilingWidths, 16, TilingArrangement.RANDOM_JITTER);
//		
//		/************************  IMPORTANT PARAMETRISATION  *************************/
////		alpha = 0.1/(tileCoding.features(initialState).size());//empirical tuning of alpha for tile coding
//		/************************************  END  ***********************************/
//		
//		//parametrisation of the source VFA following the previously obtained params
////		LinearVFA sourceVFA = new LinearVFA(tileCoding);
////		ArrayList<Double> parameters = fileToArray(sourcesPath);
////		for(int i = 0; i < parameters.size(); ++i)
////			sourceVFA.setParameter(i, parameters.get(i));
//		
//		/************************  TRY  *************************/
//		alpha = 0.1/(source.features(initialState).size());//empirical tuning of alpha for tile coding
//		
//		
//		/************************  END  *************************/
//		
//		ArrayList<DifferentiableStateActionValue> sourceVFAs = new ArrayList<DifferentiableStateActionValue>();
//		sourceVFAs.add(sourceVFA);
//		
//		//#1 returns always -1 #3 is 0 for Look-Ahead advice
//		ExtendedShapingRewardFunction advicedRewardFunction = ExtendedShapingRewardFunction.create(new UniformCostRF(), sourceVFAs, false, gamma);
//		
//		GradientDescentSarsaLamTransfer agent = new GradientDescentSarsaLamTransfer(domain, gamma, new LinearVFA(tileCoding), alpha, lambda, epsilon, advicedRewardFunction.getAdviceFunction());
//		advicedRewardFunction.setNextAction(new NextAction(agent));
//		
//		String writeString = "TL_";
//		
//		//epochs??
//		//LearningAgentFactory factory;
//		//factory = TransferAgentFactories.getGradientDescentSarsaLamTransferFactory(domain, vfa, rewardFunc, alpha, epsilon, lambda, gamma);
//		QProvider qp = (QProvider)agent;
//		
//		//run learning for 40.000 episodes
//		for(long i = 0; i < 600; i++){
//			
//			//write action values
//			osAV.print(i+ " ");
//			List<QValue> initQValues = qp.qValues(initialState); //S0
//			for(ListIterator<QValue> itr = initQValues.listIterator(); itr.hasNext() ;){
//				QValue currentQValue = itr.next();
//				double actionValue = currentQValue.q;
//				
//				osAV.print(actionValue+ " ");
//			}
//			osAV.println();
//			
//			Episode e = agent.runLearningEpisode(env,50);//object for storing the episode at each loop
//			
//			e.write(outputPath + writeString + "_" + i);			
//			
//			osR.println(i + " " + e.discountedReturn(1));
//			
//			//agent.writeQTable(outputPath+"/TIME_END.txt");mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
//			
//			//reset environment for next learning episodeQLearning a = 
//			env.resetEnvironment();
//		}
//		
//		osAV.close();
//		osR.close();
//		osNLR.close();
//		
//	}

	
	//java passes the function inputs by values so we actually get just copies of the original variables
	public void PerformCurrentPolicy(long ix, QLearning agentCopy, PrintWriter os){
		//modify the agent for fully not learning behaviours 
		agentCopy.setLearningPolicy(new GreedyDeterministicQPolicy(agentCopy));
		agentCopy.setLearningRateFunction(new ConstantLR(0.0)); // alpha=0
		
		Episode e = agentCopy.runLearningEpisode(env,50);//object for storing the episode at each loop
		os.println(ix + " " + e.discountedReturn(0.999));
		
		env.resetEnvironment();
	}
	
	public ArrayList<Double> fileToArray(String sourceFilePath){
		ArrayList<Double> toReturn = new ArrayList<Double>(); 
		
		Scanner s = null;
		try {
			s = new Scanner(new File(sourceFilePath));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		String temp = "";
		while(s.hasNext()){
			temp = s.next();
			temp = temp.replace("[", "").replace("]", "").replace(",", "");
			double d = Double.parseDouble(temp);
			toReturn.add(d);
		}
		
		return toReturn;
	}
	
	public void manualValueFunctionVis(ValueFunction valueFunction/**, Policy p*/){
		
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
		
		/**
		//define a policy painter that uses arrow glyphs for each of the grid world actions
		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, 10), new VariableDomain(0, 10), 1, 1);

		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_NORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_SOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_EAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_WEST, new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


		//add our policy renderer to it
		gui.setSpp(spp);
		gui.setPolicy(p);
		*/
		
		//set the background color for places where states are not rendered to grey
		gui.setBgColor(Color.GRAY);

		//start it
		gui.initGUI();


	}
	
	public void valueIterationExample(String outputPath){

		Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 100);
		//Policy p = planner.planFromState(initialState);

		//PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");

		manualValueFunctionVis((ValueFunction)planner/**, p*/);

	}
	
	public static void main(String[] args) {
		System.out.println("Welcome to BlockDude Learning!");
		
		BlockDudeLearning example = new BlockDudeLearning();
		String outputPathSource = "output/"; //directory to record results
		String outputPathTarget = "outputTarget/"; //directory to record results
		String sourcesPath = "source_task/Sources.txt";
		
		System.out.println("Starting Learning phase...");
		
		//block for the creation of an empty file in order to
		//keep track of the actual time for the learning start
		PrintWriter timeFile = null;
		try {
			timeFile = new PrintWriter(outputPathSource+"/TIME_BEGIN.txt");
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		timeFile.close();
		
		//learning algorithms here
		example.PerformLearningAlgorithm(outputPathSource,1);
		
		//run the visualizer
		example.visualize(outputPathSource);
		
		//Transfer Learning
		//example.PerformTransferLearningAlgorithm(outputPathTarget, sourcesPath, source);
		
		//for visualization
		//example.valueIterationExample(outputPath);
		
		//run the visualizer
		//example.visualize(outputPathTarget);
		
		return;
	}
	
	class OurLinearVFA extends LinearVFA{
		
		public Set<Integer> getKeys(){
			return weights.keySet();
		}
		
		public OurLinearVFA(SparseStateFeatures sparseStateFeatures) {
			super(sparseStateFeatures);
			// TODO Auto-generated constructor stub
		}
		
	};
	
}
