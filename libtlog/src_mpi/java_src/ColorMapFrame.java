// $Id: ColorMapFrame.java,v 1.1.1.1 2005/06/20 09:56:18 msato Exp $
// $RWC_Release$
// $RWC_Copyright$
import java.awt.*;
import java.util.*;
import java.awt.event.*;
import java.applet.Applet;

//
// Dialog for setting condition
//
class ColorMapFrame extends Frame implements ActionListener {
  Button hideButton;

  // constructor
  public ColorMapFrame(){
    setTitle("ColorMap");  // set title of this window 
    GridBagLayout layout = new GridBagLayout();
    setLayout(layout);
    
    GridBagConstraints c = new GridBagConstraints();
    c.gridwidth = GridBagConstraints.REMAINDER;
    c.anchor = GridBagConstraints.CENTER;
    
    Label label = new Label("Color Map for tlog file  ...");
    layout.setConstraints(label,c);
    add(label);
    
    for(int i = 0; i < tlogData.type_max; i++){
      switch(tlogData.type_kind[i]){
      case tlogData.TYPE_IN:
	addLegend(layout,tlogData.type_name[i],tlogData.type_color[i],true);
	break;
      case tlogData.TYPE_EVENT:
	addLegend(layout,tlogData.type_name[i],tlogData.type_color[i],false);
	break;
      }
    }

    hideButton = new Button(" Hide ");
    layout.setConstraints(hideButton,c);
    hideButton.addActionListener(this);
    add(hideButton);
    pack();
  }
  
  void addLegend(GridBagLayout layout,String name,
		 Color color,boolean isRect){
    GridBagConstraints c = new GridBagConstraints();
    c.gridwidth = GridBagConstraints.RELATIVE;
    c.anchor = GridBagConstraints.WEST;
    c.ipadx = 5;
    c.ipady = 5;
    c.insets = new Insets(2,5,2,5);
    ColorLegend lg = new ColorLegend(color,isRect);
    layout.setConstraints(lg,c);
    add(lg);

    c.gridwidth = GridBagConstraints.REMAINDER;
    Label label = new Label(name);
    layout.setConstraints(label,c);
    add(label);
  }

  public void actionPerformed(ActionEvent e){
    if(e.getSource() == hideButton){
      dispose();
    }
  }
}

class ColorLegend extends Canvas {
  boolean isRect;
  Color color;

  final static int HEIGHT = 20;
  final static int WIDTH = 30;

  public ColorLegend(Color color,boolean isRect){
    this.color = color;
    this.isRect = isRect;
  }

  public Dimension getMinimumSize(){
    return new Dimension(WIDTH,HEIGHT);
  }

  public Dimension getPreferredSize() { 
    return getMinimumSize();
  }

  public Insets getInsets() {
    return new Insets(2,5,2,5);
  }
  
  public void paint(Graphics g){
    g.setColor(color);
    if(isRect)
      g.fillRect(0,0,WIDTH,HEIGHT);
    else
      g.drawLine(WIDTH/2,0,WIDTH/2,HEIGHT);
  }
}









