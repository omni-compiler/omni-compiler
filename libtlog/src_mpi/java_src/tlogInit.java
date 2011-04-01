// $Id: tlogInit.java,v 1.1.1.1 2005/06/20 09:56:18 msato Exp $
// $Release$
// $Copyright$

import java.awt.*;

public class tlogInit {

  final static byte TLOG_EVENT_1_IN = 10;
  final static byte TLOG_EVENT_1_OUT = 11;
  final static byte TLOG_EVENT_2_IN = 12;
  final static byte TLOG_EVENT_2_OUT = 13;
  final static byte TLOG_EVENT_3_IN = 14;
  final static byte TLOG_EVENT_3_OUT = 15;
  final static byte TLOG_EVENT_4_IN = 16;
  final static byte TLOG_EVENT_4_OUT = 17;
  final static byte TLOG_EVENT_5_IN = 18;
  final static byte TLOG_EVENT_5_OUT = 19;
  final static byte TLOG_EVENT_6_IN = 20;
  final static byte TLOG_EVENT_6_OUT = 21;
  final static byte TLOG_EVENT_7_IN = 22;
  final static byte TLOG_EVENT_7_OUT = 23;
  final static byte TLOG_EVENT_8_IN = 24;
  final static byte TLOG_EVENT_8_OUT = 25;
  final static byte TLOG_EVENT_9_IN = 26;
  final static byte TLOG_EVENT_9_OUT = 27;

  final static byte TLOG_EVENT_1 = 31;
  final static byte TLOG_EVENT_2 = 32;
  final static byte TLOG_EVENT_3 = 33;
  final static byte TLOG_EVENT_4 = 34;
  final static byte TLOG_EVENT_5 = 35;
  final static byte TLOG_EVENT_6 = 36;
  final static byte TLOG_EVENT_7 = 37;
  final static byte TLOG_EVENT_8 = 38;
  final static byte TLOG_EVENT_9 = 39;

  public static void init() {
    tlogData.setMaxType(40);

    tlogData.defineEvent2Type(TLOG_EVENT_1_IN,
			      TLOG_EVENT_1_OUT,
			      "Task",Color.white);
    tlogData.defineEvent2Type(TLOG_EVENT_2_IN,
			      TLOG_EVENT_2_OUT,
			      "Loop",Color.green);
    tlogData.defineEvent2Type(TLOG_EVENT_3_IN,
			      TLOG_EVENT_3_OUT,
			      "Reflect",Color.yellow);
    tlogData.defineEvent2Type(TLOG_EVENT_4_IN,
			      TLOG_EVENT_4_OUT,
			      "Barrier",Color.blue);
    tlogData.defineEvent2Type(TLOG_EVENT_5_IN,
			      TLOG_EVENT_5_OUT,
			      "Reduction",Color.magenta);
    tlogData.defineEvent2Type(TLOG_EVENT_6_IN,
			      TLOG_EVENT_6_OUT,
			      "Bcast",Color.orange);
    tlogData.defineEvent2Type(TLOG_EVENT_7_IN,
			      TLOG_EVENT_7_OUT,
			      "Gmove",Color.pink);
    /*    tlogData.defineEvent2Type(TLOG_EVENT_8_IN,
			      TLOG_EVENT_8_OUT,
			      "8 in/out",Color.red);
    tlogData.defineEvent2Type(TLOG_EVENT_9_IN,
			      TLOG_EVENT_9_OUT,
			      "9 in/out",Color.gray);


    tlogData.defineEventType(TLOG_EVENT_1,"1 event",Color.white);
    tlogData.defineEventType(TLOG_EVENT_2,"2 event",Color.green);
    tlogData.defineEventType(TLOG_EVENT_3,"3 event",Color.yellow);
    tlogData.defineEventType(TLOG_EVENT_4,"4 event",Color.blue);
    tlogData.defineEventType(TLOG_EVENT_5,"5 event",Color.magenta);
    tlogData.defineEventType(TLOG_EVENT_6,"6 event",Color.orange);
    tlogData.defineEventType(TLOG_EVENT_7,"7 event",Color.pink);
    tlogData.defineEventType(TLOG_EVENT_8,"8 event",Color.red);
    tlogData.defineEventType(TLOG_EVENT_9,"9 event",Color.gray);*/

  }
}


