package myProj;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import burlap.behavior.policy.EnumerablePolicy;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.SolverDerivedPolicy;
import burlap.behavior.policy.support.ActionProb;
import burlap.behavior.singleagent.MDPSolverInterface;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

public class LoggingEpsilonGreedy implements EnumerablePolicy, SolverDerivedPolicy {
	
	private EpsilonGreedy eg;
	private QProvider qp;
	private String outputPath;
	private int currEp = -1;
		
	
//	public LoggingEpsilonGreedy(double epsilon){
//		eg = new EpsilonGreedy(epsilon);
//	}
	
	public LoggingEpsilonGreedy(QProvider planner, double epsilon, String path, int epoch){
		this.eg = new EpsilonGreedy(planner, epsilon);
		this.qp = planner;
		
		this.outputPath = path + "QValues/" + epoch;
		
		//Create the directory
		File outputFile = new File(this.outputPath);
		outputFile.mkdirs();
	}
	
	@Override
	public Action action(State s) {
		
		//writer for the QValues
		PrintWriter osQV = null;
		try {
//			File epochFile = new File(this.outputPath + "/" + this.currEph);
//			if(!epochFile.exists()){
//				epochFile.mkdir();
//				osQV = new PrintWriter(new FileWriter(this.outputPath + "/" + this.currEph + "/" + this.currEp + ".txt", true));
//			}else{
//				File episodeFile = new File(this.outputPath + "/" + this.currEph + "/" + this.currEp + ".txt");
//				
//				if(episodeFile.exists()){
//					this.currEph += 1;
//					File newEphFile = new File(this.outputPath + "/" + this.currEph);
//					newEphFile.mkdir();
//				}	
//				
//				osQV = new PrintWriter(new FileWriter(this.outputPath + "/" + this.currEph + "/" + this.currEp + ".txt", true));
//								
//			}
			osQV = new PrintWriter(new FileWriter(this.outputPath + "/" + this.currEp + ".txt", true));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		osQV.print("ax:" + s.get("agent:x")
				 + "ay:" + s.get("agent:y")
				 + "ad:" + s.get("agent:dir")
				 + "ah:" + s.get("agent:holding")
				 + "			");
		
		for(QValue qValue : qp.qValues(s)){
			osQV.print(qValue.a + " " + qValue.q + "     ");
		}
		osQV.println();
		osQV.close();
		
		return this.eg.action(s);
	}

	@Override
	public double actionProb(State s, Action a) {
		return this.eg.actionProb(s, a);
	}

	@Override
	public boolean definedFor(State s) {
		return this.eg.definedFor(s);
	}

	@Override
	public void setSolver(MDPSolverInterface solver) {
		this.eg.setSolver(solver);
	}

	@Override
	public List<ActionProb> policyDistribution(State s) {
		return this.eg.policyDistribution(s);
	}
	
	public void setCurrEp(int ep){
		this.currEp = ep;
	}
	
//	public void setCurrEph(int eph){
//		this.currEph = eph;
//		
//		File outputFileEpoch = new File(outputPath + "/" + this.currEph);
//		outputFileEpoch.mkdir();
//		
//	}
	
//	private void getCurrEpoch(){
//		this.currEph = 0;
//		
//		String[] innerFiles = new File(this.outputPath).list();
//		
//		for(String s : innerFiles){
//			if(new File(this.outputPath,s).isDirectory())
//				if(s.contains("epoch")){
//					int detectedEpoch = Integer.parseInt((s.split("_"))[1]);
//					
//					if(detectedEpoch > this.currEph)
//						this.currEph = detectedEpoch;
//				}	
//		}
//		
//		if(this.currEph == -1)
//			this.currEph = 0;
//	}
}