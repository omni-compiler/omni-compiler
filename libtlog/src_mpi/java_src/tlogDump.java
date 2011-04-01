// $Id: tlogDump.java,v 1.1.1.1 2005/06/20 09:56:18 msato Exp $
// $RWC_Release$
// $RWC_Copyright$
import java.io.*;

public class tlogDump {
  final static int TLOG_BLOCK_SIZE = 1024;

  public static void main(String args[]){
    byte [] block = new byte[TLOG_BLOCK_SIZE];
    FileInputStream inFile = null;
    int nbytes = 0;

    if(args.length != 1){
      System.err.println("dump file name required...");
      System.exit(1);
    }

    try {
      inFile = new FileInputStream(args[0]);
    } catch(FileNotFoundException e){
      System.out.println("cannot open logfile");
      System.exit(1);
    }

    while(true){
      try {
	nbytes = inFile.read(block);
      } catch(IOException e){
	System.out.println("I/O exception");
	System.exit(1);
      }
      if(nbytes == -1) break;
      if(nbytes != TLOG_BLOCK_SIZE){
	System.out.println("bad log file");
	System.exit(1);
      }
      DataInputStream in = 
	new DataInputStream(new ByteArrayInputStream(block));
      
      try {
	while(true){
	  try {
	      //byte type = in.readByte();
	      //int id = (int)(in.readByte());
	      //int arg1 = (int)(in.readShort());
	       int id = (int)(in.readShort());
	       byte type = in.readByte();
	       int arg1 = (int)(in.readByte());

	    int arg2 = in.readInt();
	    double time = in.readDouble();

	    if(type == 0) continue;

	    System.out.println("type="+type+",id="+id+
			       ",arg1="+arg1+",arg2="+arg2+
			       ",time="+time);
	  } catch(EOFException e){
	    break;
	  }
	}
      } catch(IOException e){
	System.out.println("bad data");
	System.exit(1);
      } 
    }
  }
}

