// $Id: tlogPanel.java,v 1.1.1.1 2005/06/20 09:56:18 msato Exp $
// $RWC_Release$
// $RWC_Copyright$

/* event display Panel */
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;
import java.util.Stack;

public class tlogPanel extends Panel 
    implements AdjustmentListener,MouseListener,MouseMotionListener
{
  tlogDrawPanel dp;
  Scrollbar hSlider,vSlider;
  Panel scalePanel, pePanel;

  double base_time;
  double scale;
  double scale_unit;

  tlogDataFile logFile;

  final static double MIN_SCALE = 1.0e-3;
  final static double MAX_SCALE = 1.0e+10;

  final static int MAX_ZOOM = 50;
  int zoomSp = 0;
  int zoomTop = 0;
  double zoomBaseT[];
  double zoomScale[];

    int vOffset;  // vertical offset

  // constructor
  public tlogPanel(){
    setLayout(new BorderLayout());
    dp = new tlogDrawPanel(this);
    hSlider = new Scrollbar(Scrollbar.HORIZONTAL,0,10,0,100);
    vSlider = new Scrollbar(Scrollbar.VERTICAL,0,10,0,100);
    scalePanel = new ScalePanel(this);
    pePanel = new PeNumPanel(this);
    add(dp,"Center");
    add(hSlider,"South");
    add(vSlider,"East");
    add(scalePanel,"North");
    add(pePanel,"West");

    // set Listener
    hSlider.setBlockIncrement(5);
    hSlider.setUnitIncrement(5);

    vSlider.setBlockIncrement(5);
    vSlider.setUnitIncrement(5);

    hSlider.addAdjustmentListener(this);
    vSlider.addAdjustmentListener(this);

    dp.addMouseListener(this);
    dp.addMouseMotionListener(this);

    zoomBaseT = new double[MAX_ZOOM];
    zoomScale = new double[MAX_ZOOM];

    vOffset = 0;
  }

  public void displayLog(tlogDataFile file){
    this.logFile = file;
    // find the first base_time and scale 
    base_time = file.min_time;
    int w = size().width;
    double s = ((double)w)/(file.max_time - base_time);
    for(scale = MIN_SCALE; scale < MAX_SCALE; scale *= 10.0)
      if(scale > s) break;
    scale = scale/10.0;

    //    System.out.println("min="+file.min_time+
    // ",max="+file.max_time+",scale="+scale);

    zoomSp = zoomTop = 0;
    zoomBaseT[zoomSp] = base_time;
    zoomScale[zoomSp] = scale;
    updateView();
  }
  
  int TtoX(double time){ return (int)((time - base_time)*scale); }
  double XtoT(int x){ return (base_time + ((double)x)/scale); }

  /*
  public void resetScale(){
    if(client == null) return;	// no data
    
    float dx = maxX - minX;
    float dy = maxY - minY;
    
    // set origin
    orgX = minX + dx/2;
    orgY = minY + dy/2;
    float sx = ((float)dp.size().width)/dx;
    float sy = ((float)dp.size().height)/dy;
    scale = sx < sy ? sx : sy;
    scale *= 0.95;
    updateView();
  }
  */

  int sliderPos() {
    double max_time = logFile.max_time;
    double min_time = logFile.min_time;
    return (int)((base_time - min_time)*100.0/(max_time-min_time)+0.5);
  }

  /* update view with the current scale */
  public void updateView(){
    if(logFile == null) return;
    double max_time = logFile.max_time;
    double min_time = logFile.min_time;
    int w = dp.size().width;

    // update scroll bar (horizontal)
    int v = sliderPos();
    if(hSlider.getValue() != v)  hSlider.setValue(v);
    v = (int)((double)w*100.0/(scale*(max_time-min_time)));
    if(v <= 0) v = 1;
    if(v >= 50) v = 50;
    if(v != hSlider.getVisibleAmount()) hSlider.setVisibleAmount(v);

    // update scroll bar (vertical)
    int h = dp.size().height/dp.BAR_HEIGHT;
    v = (int)(100.0*h/logFile.npe);
    if(v <= 0) v = 1;
    if(v >= 100) v = 100;
    if(v != vSlider.getVisibleAmount()) vSlider.setVisibleAmount(v);
    
    scalePanel.repaint();
    pePanel.repaint();
    dp.repaint();
  }

  public void adjustmentValueChanged(AdjustmentEvent e){
    // System.out.println(e.paramString());
    if(logFile == null) return;

    if(e.getSource() == hSlider){  // horizontal slider
      switch(e.getAdjustmentType()){
      case AdjustmentEvent.UNIT_INCREMENT:
	base_time += 10.0/scale;
	break;
      case AdjustmentEvent.UNIT_DECREMENT:
	base_time -= 10.0/scale;
	break;
      default:
	int v = hSlider.getValue();
	if(sliderPos() == v) return;
	double max_time = logFile.max_time;
	double min_time = logFile.min_time;
	base_time = min_time + (max_time - min_time)*(((double)v)/100.0);
      }
      updateView();
      return;
    } 

    if(e.getSource() == vSlider){  // vertical silder
	int v = vSlider.getValue();
	vOffset = -(int)(logFile.npe*dp.BAR_HEIGHT*((double)v)/100.0);
	// System.out.println("vOffset="+vOffset);
	updateView();
	return;
    }
  }

  public void zoomUp(){
    if(logFile == null) return;	// no data
    if(scale >= MAX_SCALE) return;
    if(zoomSp >= (MAX_ZOOM-1)) return;
    zoomSp++;
    if(zoomTop >= zoomSp){
      base_time = zoomBaseT[zoomSp];
      scale = zoomScale[zoomSp];
    } else {
      zoomTop = zoomSp;
      scale *= 10.0;
      zoomBaseT[zoomSp] = base_time;
      zoomScale[zoomSp] = scale;
    }
    updateView();
  }
  
  public void zoomDown(){
    if(logFile == null) return;	// no data
    if(scale <= MIN_SCALE) return;
    if(zoomSp <= 0) return;
    zoomSp--;
    base_time = zoomBaseT[zoomSp];
    scale = zoomScale[zoomSp];
    updateView();
  }

  /*
   * handling mouse event 
   */
  public void mousePressed(MouseEvent e){
    // area.rect = new Rectangle(e.getX(),e.getY(),0,0);
    // area.repaint();
    // System.out.println("T="+XtoT(e.getX()));
  }

  public void mouseReleased(MouseEvent e){
    if(dp.dragRect == null) return;
    int x1 = dp.dragRect.x;	// start point
    int x2 = e.getX();		// end point
    dp.dragRect = null;
    if(x2 <= x1){
      dp.repaint();
    } else {
      double s = ((double)dp.size().width)/(XtoT(x2) - XtoT(x1));
      double t = XtoT(x1);
      if(s >= MAX_SCALE || zoomSp >= (MAX_ZOOM-1)){
	updateView();
	return;
      }
      scale = s;
      base_time = t;
      zoomSp++;
      zoomBaseT[zoomSp] = base_time;
      zoomScale[zoomSp] = scale;
      zoomTop = zoomSp;
      updateView();
    }
  }

  public void mouseClicked(MouseEvent e){ }
  public void mouseEntered(MouseEvent e){ }
  public void mouseExited(MouseEvent e){ }

  public void mouseDragged(MouseEvent e){
    if(dp.dragRect == null){
      dp.dragRect = new Rectangle(e.getX(),e.getY(),0,0);
    }
    dp.dragRect.setSize(e.getX()-dp.dragRect.x,e.getY()-dp.dragRect.y);
    dp.repaint();
  }

  public void mouseMoved(MouseEvent e){ }
}


/*
 * panels
 */
class tlogDrawPanel extends Panel {
  tlogPanel parent;
    
  Image sub_im;
  Graphics sub_g;	// sub image

  Color fgc = Color.white;
  Color bgc = Color.black;

  final static int INIT_WIDTH = 600;
  final static int INIT_HEIGHT = 400;

  public final static int BAR_HEIGHT = 10;
  public final static int EVENT_BAR_HEIGHT = 5;

  final static int N_NEST = 100;
  final static int MAX_NEST = 5;
  final static int NEST_HEIGHT = BAR_HEIGHT/MAX_NEST;
  tlogData nest_stack[];

  Rectangle dragRect;
  Color dragColor = Color.red;

  public tlogDrawPanel(tlogPanel parent){
    setBackground(bgc);
    setForeground(fgc);
    this.parent = parent;
    nest_stack = new tlogData [N_NEST];
  }

  public Dimension getPreferredSize(){
    return getMinimumSize();
  }

  public Dimension getMinimumSize(){
    return new Dimension(INIT_WIDTH,INIT_HEIGHT
			 /* BAR_HEIGHT*parent.logFile.npe*/);
  }

  // for debug
  //  public void repaint(){
  //    System.out.println("repaint Draw ...");
  //    super.repaint();
  //    Thread.dumpStack();
  //  }
  

  // clear sub image
  public void reshape(int x, int y, int width, int height){
    sub_g = null;
    super.reshape(x, y, width, height);
  }

  public void update(Graphics g){
    boolean new_img;
    if(sub_g == null){
      new_img = true;
      sub_im = this.createImage(size().width, size().height);
      sub_g = sub_im.getGraphics();
    } else new_img = false;

    if(dragRect == null || new_img){
      sub_g.setColor(bgc);
      sub_g.fillRect(0, 0, size().width, size().height);
      sub_g.setColor(getForeground());
      paint(sub_g);	// replaint for subg
      g.drawImage(sub_im, 0, 0, this);  // show image on display!!!
    } 

    if(dragRect != null){
      g.drawImage(sub_im, 0, 0, this);
      g.setColor(dragColor);
      g.drawRect(dragRect.x,dragRect.y,dragRect.width-1,dragRect.height-1);
    }
  }
    
  /*
   * main paint routine 
   */
  public void paint(Graphics g){
    tlogDataFile log = parent.logFile;
    if(log == null) return; // do nothing, if data is not set.

    g.translate(0,parent.vOffset);

    for(int pe = 0; pe < log.npe; pe++){
      int y = pe*BAR_HEIGHT;
      tlogData d,dd;
      int last_x,x;

      Vector logs = log.logData[pe];
      if(logs.isEmpty()) continue;	// empty events 

      int start = log.dispHint[pe];
      int end = log.logData[pe].size();

      // search start point.
      d = (tlogData) logs.elementAt(start);
      x = parent.TtoX(d.timestamp);
      // System.out.println("start d="+d+",x="+x);

      if(x < 0){
	// search forward
	for( ; start < end; start++){
	  d = (tlogData) logs.elementAt(start);
	  if(parent.TtoX(d.timestamp) >= 0) break;
	}
	if(start > 0) start--;
      } else {
	// search backward
	for( ; start > 0; start--){
	  d = (tlogData) logs.elementAt(start);
	  if(parent.TtoX(d.timestamp) < 0) break;
	}
      }

      d = (tlogData) logs.elementAt(start);
      last_x = parent.TtoX(d.timestamp);
      // System.out.println("last d="+d+",last_x="+last_x);

      // setup nested stack.
      int nest_level = 0;
      d = (tlogData) logs.elementAt(start);
      Stack s = new Stack();
      for(dd = d.getNested(); dd != null; dd = dd.getNested())
	s.push(dd);
      while(!s.empty()){
	if(nest_level >= N_NEST) break;
	nest_stack[nest_level++] = (tlogData) s.pop();
      }

      if(tlogData.type_kind[d.type] == tlogData.TYPE_IN){
	if(nest_level >= N_NEST) break;
	nest_stack[nest_level++] = d;
      } else if(tlogData.type_kind[d.type] == tlogData.TYPE_OUT){
	if(nest_level > 0) nest_level--;
      }

      for(int i = start+1; i < end; i++){
	d = (tlogData) logs.elementAt(i);
	x = parent.TtoX(d.timestamp);

	// System.out.println("d="+d+",x="+x+",next_level="+nest_level);

	if(nest_level > 0){
	  for(int lev = 0; lev < nest_level; lev++){
	    if(lev >= MAX_NEST) break;
	    dd = nest_stack[lev];
	    g.setColor(tlogData.type_color[dd.type]);
	    g.fillRect(last_x,y,x-last_x,BAR_HEIGHT-1-lev*NEST_HEIGHT);
	    // System.out.println("rect last_x="+last_x+",x="+x);
	  }
	}
	if(x > size().width) break;
	last_x = x;

	switch(tlogData.type_kind[d.type]){
	case tlogData.TYPE_EVENT:
	  g.setColor(tlogData.type_color[d.type]);
	  g.drawLine(x-1,y,x-1,y+EVENT_BAR_HEIGHT);
	  break;

	case tlogData.TYPE_IN:
	  if(nest_level >= N_NEST) break; /* nest stack overflow */
	  nest_stack[nest_level++] = d;
	  break;

	case tlogData.TYPE_OUT:
	  if(nest_level <= 0) break;
	  nest_level--;
	}
      }
    }
  }
} 

//
// draw scale ruler
//
class ScalePanel extends Panel {
  final static int HEIGHT = 30;
  Color fgc = Color.black;
  Color bgc = Color.white;
  tlogPanel parent;

  public ScalePanel(tlogPanel parent){
    //    super();
    setBackground(bgc);
    setForeground(fgc);
    this.parent = parent;
  }

  public Dimension getPreferredSize(){
    return getMinimumSize();
  }
  public Dimension getMinimumSize(){
    return new Dimension(0,HEIGHT);
  }

  public void paint(Graphics g){
    int w = size().width;
    g.setColor(fgc);

    double t,t_unit;
    t = parent.XtoT(50) - parent.XtoT(0); // minmum space, 50 pixel
    for(t_unit = 1000.0; t_unit > t; t_unit *= 0.1) /* */;
    t_unit *= 10.0;
    //System.out.println("t="+t+",t_unit="+t_unit);
    //System.out.println("x_gap="+(parent.TtoX(t_unit*2)-parent.TtoX(t_unit)));
    /* if gap is 100 pixel, change 0.5 pitch */
    if((parent.TtoX(t_unit*2)-parent.TtoX(t_unit)) > 100)
      t_unit /= 2.0;

    t = parent.base_time;
    t = ((int)(t/t_unit))*t_unit;
    while(true){
      int x = parent.TtoX(t) + PeNumPanel.WIDTH;
      if(x > w) break;
      g.drawString(Float.toString((float)t),x-5,HEIGHT/2);
      g.drawLine(x,HEIGHT/2,x,HEIGHT);
      t += t_unit;
    }
  }
}

class PeNumPanel extends Panel {
  final static int WIDTH = 50;
  Color fgc = Color.black;
  Color bgc = Color.white;
  tlogPanel parent;
    Font font;

  public PeNumPanel(tlogPanel parent){
    //    super();
    setBackground(bgc);
    setForeground(fgc);
    this.parent = parent;
    this.font = new Font("Courier",Font.BOLD,10);
  }

  public Dimension getPreferredSize(){
    return getMinimumSize();
  }

  public Dimension getMinimumSize(){
    return new Dimension(WIDTH,0);
  }
  
  public void paint(Graphics g){
    tlogDataFile log = parent.logFile;
    if(log == null) return; // do nothing, if data is not set.
    g.translate(0,parent.vOffset);
    g.setFont(font);
    FontMetrics fm = g.getFontMetrics();
    g.setColor(fgc);
    for(int pe = 0; pe < log.npe; pe++){
      int y = pe*tlogDrawPanel.BAR_HEIGHT+fm.getAscent();
      g.drawString(" "+pe,5,y);
    }
    // g.translate(0,-parent.vOffset);
  }
}

