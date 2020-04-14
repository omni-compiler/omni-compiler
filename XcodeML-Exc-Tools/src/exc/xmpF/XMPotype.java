package exc.xmpF;

import java.io.Serializable;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

import java.util.HashSet;
import java.util.HashMap;
import exc.object.*;

public class XMPotype implements Serializable {
    public HashSet<String> wr_a;
    public HashSet<String> rd_a;
    public HashSet<String> wr_v;
    public String aligned;
    public HashMap<String,XobjList> expander;
    //
    public HashSet<String> reflected;
    public HashSet<String> first_rd_a;
    XMPotype (HashSet<String> wr_a,
	   HashSet<String> rd_a,
	   HashSet<String> wr_v,
	   String aligned,
	   HashMap<String,XobjList> expander,
	   //
	   HashSet<String> reflected,
	   HashSet<String> first_rd_a
	   ) {
	this.wr_a = wr_a;
	this.rd_a = rd_a;
	this.wr_v = wr_v;
	this.aligned = aligned;
	this.expander = expander;
	//
	this.reflected = reflected;
	this.first_rd_a = first_rd_a;
    }
}
