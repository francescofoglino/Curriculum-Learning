package GridWorld;

import TransferLearning.*;
import BlockDude.BlockDudeTransferLearning_UTILS;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

import TransferLearning.TransferLearning;
import burlap.behavior.functionapproximation.sparse.tilecoding.TileCodingFeatures;
import burlap.behavior.learningrate.ConstantLR;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.valuefunction.QProvider;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldDomain.AtLocationPF;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import burlap.domain.singleagent.gridworld.state.GridLocation;

public class GridWorldTransferLearning implements TransferLearning{
	
	/** GridWorldTransferLearning variables */
	GridWorldDomain gw;
	OOSADomain domain;
	RewardFunction rf;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	String refTag;
	int nTilings;
	/** ************************************ */
	
	public GridWorldTransferLearning(GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf, int numTilings) {
		
//		int [][] currMap = state0.map.map;
//		int x = currMap.length;
//		int y = currMap[0].length;
		refTag = gwdf.tag;
 		
		GridWorldState state0 = gwdf.generateInitialState();
		
		tf = new GridWorldTerminalFunction(((Integer)state0.get("treasure:x")), ((Integer)state0.get("treasure:y")));
		
		for(int p = 1; p <= gwdf.pits.size(); ++p)//set all the pits to terminal states
			((GridWorldTerminalFunction)tf).markAsTerminalPosition(((Integer)state0.get("pit" + p + ":x")), ((Integer)state0.get("pit" + p + ":y")));
		
		gw = new GridWorldDomain(gwdf.map);
		gw.setTf(tf);
		
		gw.setRf(new GridWorldRF(gw, gwdf));
		
		goalCondition = new TFGoalCondition(tf);
		
		domain = gw.generateDomain();
		
		initialState = state0;
		
		//instantiate the HashableStateFactory
		hashingFactory = new SimpleHashableStateFactory();
		
		//creation of the actual environment with which our agent will interact
		env = new SimulatedEnvironment(domain, initialState);
		
		nTilings = numTilings;
		
		/** TESTING MAPS */
//		Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
//		VisualExplorer exp = new VisualExplorer(domain, v, initialState);
//		
//		//use w-s-a-d-x
//		exp.addKeyAction("w", GridWorldDomain.ACTION_NORTH, "");
//		exp.addKeyAction("s", GridWorldDomain.ACTION_SOUTH, "");
//		exp.addKeyAction("a", GridWorldDomain.ACTION_WEST, "");
//		exp.addKeyAction("d", GridWorldDomain.ACTION_EAST, "");
//		
//		exp.initGUI();
		/** END */
	}
	
	public TLLinearVFA PerformLearningAlgorithm(String outputPath, int algorithmChoice, int nEpochs, int nEpisodes, TLLinearVFA...sourceVFAs) {
		
		System.out.println(outputPath);
		
		File outputFile = new File(outputPath);
		outputFile.mkdirs();
		
		TLLinearVFA vfa = null;
		
		State stateToPrint = initialState;
		
		/** Declaration of output files of diverse purposes */
		//writer for the Action-Value of a particular state
		PrintWriter osAV = null;
		try {
			osAV = new PrintWriter(outputPath+"/ActionValueFunction.txt");
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		
		//writer for the Return
		PrintWriter osR = null;
		try {
			osR = new PrintWriter(outputPath+"/Return.txt");
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		/** *********************************************** */
		
		for(int epoch=0; epoch<nEpochs; ++epoch){	
			
			System.out.print(epoch);
			
			//Learning Parameters
			double gamma = 0.999;
			double alpha = 0.1;
			double alphaTilingWise = 0;//this is the actual alpha you want to associate with the tilings in order to spread the general alpha
			double lambda = 0.9;
			double qInit = 0.; //Q-values initialisation
			
			/** to change */
			GradientDescentSarsaLam agent = null;
			String writeString = "";
			
			if(algorithmChoice == 1) {
//				System.out.println("You have chosen Q-learning for Transfer Learning");
//				
//				agent = new QLearning(domain, gamma, hashingFactory, qInit, alpha);
				
			}else if(algorithmChoice == 2){
				//System.out.println("You have chosen Sarsa(lambda) for Transfer Learning");
				
				TLTileCodingFeatures tileCoding = GridWorldTransferLearning_UTILS.CreateTransferLearningTCFeatures(refTag, nTilings);
				vfa = new TLLinearVFA(tileCoding);
				
				//All the Transfer happens here
				if(sourceVFAs != null){
					//These two lines fix the problem with the input sourceVFAs when passed as a variable X = null
					if(sourceVFAs.length == 1 && sourceVFAs[0] == null)
						sourceVFAs = null;
					else{
					lambda = 0.9;
					
					for(TLLinearVFA sVFA : sourceVFAs)//loops over all the sources for transferring knowledge in parallel
						sVFA.transferVFA(vfa, null);
					
					//stateToPrint = BlockDudeTransferLearning_UTILS.initialState_2_hard_firstStep;
					}
				}
				
				alphaTilingWise = alpha/((TLTileCodingFeatures)vfa.sparseStateFeatures).numTIlings();
				
				agent = new GradientDescentSarsaLam(domain, gamma, vfa, alphaTilingWise, lambda);
				
				//to comment in case of checking the optimal target policy trajectory
//				if(sourceVFAs != null){
//					GridWorldTransferLearning_UTILS.PerformLearntPolicy(agent, env, outputPath, "transferredPolicy", -1);
//					agent.setLearningRate( new ConstantLR(alphaTilingWise));
//				}	
				
				writeString = "TL_";
				
				stateToPrint = initialState;
			}else{
				System.out.println("Your algorithm choice (" + writeString + ") is not available. "
						+ "Please select one of the followings:"
						+ "\n - 0 for Q-Learning"
						+ "\n - 1 for Sarsa(lambda)");
				return null;
			}
			
			QProvider qp = (QProvider)agent;
			
			double epsilon = 0.1;//starting epsilon value from which we start lowering it
//			LoggingEpsilonGreedy lEpsGrdy = new LoggingEpsilonGreedy(agent, epsilon, outputPath, epoch);
//			agent.setLearningPolicy(lEpsGrdy);
			EpsilonGreedy lEpsGrdy = new EpsilonGreedy(agent, epsilon);
			agent.setLearningPolicy(lEpsGrdy);
			
			//one loop for each episode
			for(long episode = 0; episode < nEpisodes; episode++){
				
				if(outputPath != null)
					GridWorldTransferLearning_UTILS.WriteActionValues(osAV, qp, stateToPrint, episode);
				
				/** kill exploration after a specified number of episodes */
				double explorationRatio = 15./20;//the fraction of episodes we do with normal exploration
				double lowerExplorationRatio = 4./20;//the fraction of episodes we do lowering exploration
				
				double stopExp = (explorationRatio+lowerExplorationRatio)*nEpisodes;//the episode when we stop the exploration
				double startExp = nEpisodes*explorationRatio; //when the lowering starts to be applied
				double intervalEp = stopExp - startExp;  //interval of episodes where we decrease the value of epsilon
				
				if(episode>=(startExp) && episode <= stopExp){
					epsilon = epsilon * (stopExp - episode)/intervalEp;
					
					if(agent instanceof GradientDescentSarsaLam){
						GradientDescentSarsaLam agentQLearning = (GradientDescentSarsaLam)agent;
//						LoggingEpsilonGreedy newLEpsGrdy = new LoggingEpsilonGreedy(agentQLearning, epsilon, outputPath, epoch);
//						agentQLearning.setLearningPolicy(newLEpsGrdy);
						
//						newLEpsGrdy.setCurrEp((int)episode);
						
						EpsilonGreedy newLEpsGrdy = new EpsilonGreedy(agent, epsilon);
						agent.setLearningPolicy(newLEpsGrdy);
					}else{
						System.out.println("ERROR! You are trying to change the learning policy to an invalid agent!");
					}
				}else if(episode > stopExp){
					epsilon = 0.;
					
					GradientDescentSarsaLam agentQLearning = (GradientDescentSarsaLam)agent;
//					LoggingEpsilonGreedy newLEpsGrdy = new LoggingEpsilonGreedy(agentQLearning, epsilon, outputPath, epoch);
//					agentQLearning.setLearningPolicy(newLEpsGrdy);
					
//					newLEpsGrdy.setCurrEp((int)episode);
					
					EpsilonGreedy newLEpsGrdy = new EpsilonGreedy(agent, epsilon);
					agent.setLearningPolicy(newLEpsGrdy);
				}else {}
//					lEpsGrdy.setCurrEp((int)episode);
				
				//System.out.println(episode + "," + epsilon + ";");//epsilon check
				/** ***************************************************** */	
				
				Episode e = agent.runLearningEpisode(env,50);//object for storing the episode at each loop
				
				//PRINT EPISODE BY EPISODE
//				e.write(outputPath + writeString + "_" + epoch + "_" + episode);			
				
				if(outputPath != null)
					osR.println(episode + " " + e.discountedReturn(1) + " " + agent.getLastNumSteps());
				
				//reset environment for next learning
				env.resetEnvironment();
				
				if(episode % 10 == 0 || episode == nEpisodes-1){
					
					((TLTileCodingFeatures)vfa.sparseStateFeatures).setLearning(false);
					
					BlockDudeTransferLearning_UTILS.PerformLearntPolicy(agent, env, outputPath, "policyEvaluation.txt", episode);
					agent.setLearningRate( new ConstantLR(alphaTilingWise));
					agent.setLearningPolicy(lEpsGrdy);
					//agent.setLearningPolicy(new EpsilonGreedy(epsilon));
					
					((TLTileCodingFeatures)vfa.sparseStateFeatures).setLearning(true);
				}
				
//				if(episode == nEpisodes-1)
//					GridWorldTransferLearning_UTILS.PerformLearntPolicy(agent, env, outputPath, "finalPolicy", -1);
			}
			
//			if(vfa.sparseStateFeatures instanceof TLTileCodingFeatures)
//				System.out.println(((TLTileCodingFeatures)vfa.sparseStateFeatures).getTransferCovering());

		}
		
		osAV.close();
		osR.close();
		
		return vfa;
	}
	
	/** tool for visualising the result of the learning */
	public void visualize(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gw.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputPath);
	}
	/** *********************************************** */
	
	/** REWARD FUNCTION */
	public class GridWorldRF implements RewardFunction{
		
		GridWorldDomain domain;
		GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf;
		
		GridWorldRF(GridWorldDomain domain, GridWorldTransferLearning_UTILS.GridWorldDomainFeatures gwdf){
			this.domain = domain;
			this.gwdf = gwdf;
		};
		
		@Override
		public double reward(State s, Action a, State sprime) {
			AtLocationPF al = domain.new AtLocationPF(GridWorldDomain.CLASS_LOCATION, new String[]{GridWorldDomain.CLASS_AGENT});
			
			//((Integer)state0.get("treasure:x")), ((Integer)state0.get("treasure:y"));
			
			if(al.isTrue(((OOState)sprime), new String[]{GridWorldDomain.CLASS_AGENT,"treasure"}))//if the current action will lead me to the goal
				return 200;
			else {

				for(int p = 1; p <= gwdf.pits.size(); ++p) {//if the current action will lead us to a pit
					if(al.isTrue(((OOState)sprime), new String[]{GridWorldDomain.CLASS_AGENT,"pit" + p}))
						return -2500;
				}
				

				boolean onFire = false, nextToFire = false;
				
				for(int f = 1; f <= gwdf.fires.size(); ++f) {					
					if(al.isTrue(((OOState)sprime), new String[]{GridWorldDomain.CLASS_AGENT,"fire" + f}))
						onFire = true; //-500;
					else if(al.isTrue(((OOState)s), new String[]{GridWorldDomain.CLASS_AGENT,"fire" + f}) || manhattanDistance((GridWorldState)sprime, f) == 1)//from inside a fire whichever action I take leads me in a "next to fire" state
						nextToFire = true; //-250;
				}
				
				if(onFire)
					return -500;
				else if(nextToFire)
					return -250;
				
				return -1;
			}
			
		}
		
		private int manhattanDistance(GridWorldState state, int i) {
			//Vector<Integer> vctr = new Vector<Integer>();
			
			int dx = Math.abs(((Integer)state.get("fire" + i + ":x")) - ((GridWorldState)state).agent.x);
			int dy = Math.abs(((Integer)state.get("fire" + i + ":y")) - ((GridWorldState)state).agent.y);
			
			return dx + dy;
		}
	}	
	/** END */
	
	public static void main(String[] args) {
		
		System.out.println("Welcome to GridWorldTransferLearning!");
		
		//SOURCE
//		GridWorldTransferLearning source = new GridWorldTransferLearning(new GridWorldTransferLearning_UTILS().gwdf_treasure_v2, 4);
//		
//		int nEpochs_source = 1;
//		int nEpisodes_source = 100;
//		
//		TLLinearVFA sourceVFA =  source.PerformLearningAlgorithm("GridWorldTest/source/", 2, nEpochs_source, nEpisodes_source, null);
//		
//		source.visualize("GridWorldTest/source/");
//		
//		//SOURCE 2
//		GridWorldTransferLearning source2 = new GridWorldTransferLearning(new GridWorldTransferLearning_UTILS().gwdf_0_simplified, 4);
//		
//		int nEpochs_source2 = 1;
//		int nEpisodes_source2 = 2000;
//		
//		TLLinearVFA source2VFA =  source2.PerformLearningAlgorithm("GridWorldTest/source2/", 2, nEpochs_source2, nEpisodes_source2, sourceVFA);
//		
//		source2.visualize("GridWorldTest/source2/");
//		
//		//TARGET
//		GridWorldTransferLearning target = new GridWorldTransferLearning(new GridWorldTransferLearning_UTILS().gwdf_0, 4);
//		
//		int nEpochs_target = 1;
//		int nEpisodes_target = 5000;
//		
//		TLLinearVFA targetVFA =  target.PerformLearningAlgorithm("GridWorldTest/target/", 2, nEpochs_target, nEpisodes_target, source2VFA);
//		
//		target.visualize("GridWorldTest/target/");
		
		//TARGET NO TRANSFER
		GridWorldTransferLearning targetNoTransfer = new GridWorldTransferLearning(new GridWorldTransferLearning_UTILS().gwdf_test_Lv2, 4);
		
		int nEpochs_targetNoTransfer = 10;
		int nEpisodes_targetNoTransfer = 300;
		
		TLLinearVFA targetVFANoTransfer =  targetNoTransfer.PerformLearningAlgorithm("../../../temp/otherTargets/exp_3/targets/test_Lv2/", 2, nEpochs_targetNoTransfer, nEpisodes_targetNoTransfer, null);
		
		targetNoTransfer.visualize("../../../temp/otherTargets/exp_3/targets/test_Lv2/");
	}

}
