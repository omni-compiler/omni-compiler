// $Id: tlogview.java,v 1.1.1.1 2005/06/20 09:56:18 msato Exp $
// $RWC_Release$
// $RWC_Copyright$

/*
 *  tlog viewer
 */

import java.awt.*;
import java.applet.*;
import java.io.*;

public class tlogview extends Applet {
  boolean isApplet = true;
  tlogPanel ep;
  Button zoomUp, zoomDown, ShowColorMap, Exit;
  TextField fname_f;
  tlogDataFile file;  
  ColorMapFrame ColorMap;

  public void init(){
    setLayout(new BorderLayout());
    Panel buttonPanel = new Panel(new FlowLayout(FlowLayout.LEFT));
    zoomUp = new Button("Zoom Up");
    zoomDown = new Button("Zoom Down");
    Exit = new Button("Exit");
    ShowColorMap = new Button("ShowColorMap");

    fname_f = new TextField(30);
    fname_f.setEditable(false);
    buttonPanel.add(Exit);
    buttonPanel.add(zoomUp);
    buttonPanel.add(zoomDown);
    buttonPanel.add(fname_f);
    buttonPanel.add(ShowColorMap);
    add(buttonPanel,"North");
    ep = new tlogPanel();
    add(ep,"Center");
  }

  void setLogFile(String filename){
    fname_f.setText(filename);
    file = new tlogDataFile();
    file.Input(filename);
    ep.displayLog(file);
  }

  public boolean action(Event evt,Object arg){
    if (evt.target == zoomUp){
      ep.zoomUp();
      return true;
    } else if (evt.target == zoomDown){
      ep.zoomDown();
      return true;
    } else if(evt.target ==  ShowColorMap){
      if(ColorMap == null) ColorMap = new ColorMapFrame();
      ColorMap.show();
      return true;
    } else if (evt.target == Exit){
      System.exit(0);
      return true;
    } else return false;
  }

  public static void main(String args[]) {
    if(args.length != 1){
      System.err.println("dump file name required...");
      System.exit(1);
    }
    Frame f = new Frame("tlogview");
    // f.resize(600, 500);
    tlogview app = new tlogview();
    app.isApplet = false;
    app.init();
    f.add("Center", app);
    f.pack();
    tlogInit.init();
    app.setLogFile(args[0]);
    f.show();
  }
}



