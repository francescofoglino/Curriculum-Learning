package BlockDude;

import java.awt.AWTException;
import java.awt.Frame;
import java.awt.Robot;
import java.awt.event.KeyEvent;

import javax.swing.JFrame;

import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.visualizer.Visualizer;

public class BlockDudePlayBack {
	
	BlockDude bd;
	OOSADomain domain;
	String playBackPath;
	
	
	public BlockDudePlayBack(int x, int y, String path){
		
		this.bd = new BlockDude(x,y);
		this.domain = bd.generateDomain();
		this.playBackPath = path;
	}
	
	public void visualize(){
		Visualizer v = BlockDudeVisualizer.getVisualizer(this.bd.getMaxx(), this.bd.getMaxy());
		new EpisodeSequenceVisualizer(v, this.domain, this.playBackPath);
	}
	
	public void playAllLearning(int episodes, int actions) {
		Frame[] frames = JFrame.getFrames();
		
		Robot r = null;
		try {
			r = new Robot();
		} catch (AWTException e) {
			e.printStackTrace();
		}
		
		//wait 5 seconds before proceeding
		try {
			  Thread.sleep(20000L);
		} catch(InterruptedException ie) {}
		
		long betweenKeys  = 15;
		
		for(int e = 0; e < episodes; ++e) {
			r.keyPress(KeyEvent.VK_DOWN);
			r.keyRelease(KeyEvent.VK_DOWN);
			
			try {
				  Thread.sleep(betweenKeys);
			} catch(InterruptedException ie) {}
			
			r.keyPress(KeyEvent.VK_TAB);
			r.keyRelease(KeyEvent.VK_TAB);
			
			try {
				  Thread.sleep(betweenKeys);
			} catch(InterruptedException ie) {}
			
			for(int a = 0; a < actions; ++a) {
				r.keyPress(KeyEvent.VK_DOWN);
				r.keyRelease(KeyEvent.VK_DOWN);
				try {
					  Thread.sleep(betweenKeys);
				} catch(InterruptedException ie) {}
			}	
			
			r.keyPress(KeyEvent.VK_SHIFT);
			r.keyPress(KeyEvent.VK_TAB);
			r.keyRelease(KeyEvent.VK_SHIFT);
			r.keyRelease(KeyEvent.VK_TAB);
			
			try {
				  Thread.sleep(betweenKeys);
			} catch(InterruptedException ie) {}
		}	
	}
	
	public static void main(String[] args){
		
		int x = 10, y = 6;
		String path = "TLTest/outputTarget/";//"Curricula/Curriculum[1, 8, 10]/CurrEpoch_2/outputTarget/";//
		
		BlockDudePlayBack bdPlayBack = new BlockDudePlayBack(x, y, path);
		bdPlayBack.visualize();
		
		bdPlayBack.playAllLearning(105, 50);
		
	}

}
	