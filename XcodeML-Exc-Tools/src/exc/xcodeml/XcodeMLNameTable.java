package exc.xcodeml;

import java.util.Hashtable;
import exc.object.*;

public class XcodeMLNameTable {
	protected class XcodeMLName {
		Xcode code;
		String name;

		XcodeMLName(Xcode code, String name) {
			this.code = code;
			this.name = name;
		}
	};

	Hashtable<String, Xcode> ht;

	// constructor
	public XcodeMLNameTable() {
	}

	protected void initHTable(XcodeMLName table[]) {
		ht = new Hashtable<String, Xcode>();
		for (XcodeMLName t : table)
			ht.put(t.name, t.code);
	}

	public Xcode getXcode(String name) {
		return ht.get(name);
	}
}
