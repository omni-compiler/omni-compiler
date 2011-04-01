// $Id: tlogData.java,v 1.1.1.1 2005/06/20 09:56:18 msato Exp $
// $Release$
// $Copyright$

import java.awt.*;

public class tlogData {
  final static byte UNDEF = 0;	/* undefined */
  final static byte END = 1; 	/* END*/
  final static byte START = 2;
  final static byte RAW = 3; 	/* RAW information */

  byte type;
  int proc_id;
  int arg1,arg2;
  double timestamp;
  tlogData nested;  // for nested event
  
  public tlogData(byte type,int proc_id,int arg1,int arg2,double timestamp){
    this.type = type;
    this.proc_id = proc_id;
    this.arg1 = arg1;
    this.arg2 = arg2;
    this.timestamp = timestamp;
  }

  public void setNested(tlogData d){ nested = d; }
  public tlogData getNested() { return nested; }
      
  public String toString(){
    return "{type="+type+",id="+proc_id+",arg1="+arg1+",arg2="+arg2+
      ",time="+timestamp+"}";
  }

  // static function and data
  final static int TYPE_NONE = 0;
  final static int TYPE_IN = 1;
  final static int TYPE_OUT = 2;
  final static int TYPE_EVENT = 3;

  static int type_max;
  static int type_kind[];
  static Color type_color[];
  static String type_name[];

  static void setMaxType(int max){
    type_max = max;
    type_kind = new int[max];
    type_color = new Color[max];
    type_name = new String[max];
  }

  static void defineEventType(int type,String name,Color c){
    type_kind[type] = TYPE_EVENT;
    type_color[type] = c;
    type_name[type] = name;
  }

  static void defineEvent2Type(int type1,int type2,String name,Color c){
    type_kind[type1] = TYPE_IN;
    type_kind[type2] = TYPE_OUT;
    type_color[type1] = c;
    type_color[type2] = c;
    type_name[type1] = name;
  }
}


