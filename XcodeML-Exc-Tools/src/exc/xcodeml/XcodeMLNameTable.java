package exc.xcodeml;

import static xcodeml.util.XmLog.fatal;

import java.util.Hashtable;

import exc.object.Xcode;

public abstract class XcodeMLNameTable {
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


	protected XcodeMLName[] opcodeToNameTable;

	protected void initOpcodeToNameTable(XcodeMLName table[]) {
		Xcode[] allXcodes = Xcode.values();
		opcodeToNameTable = new XcodeMLName[allXcodes.length];
		for (XcodeMLName n : table) {
			opcodeToNameTable[n.code.ordinal()] = n;
		}
	}

	public String getName(Xcode code) {
		XcodeMLName n = opcodeToNameTable[code.ordinal()];
		if (n == null) {
			fatal("Unknown xcode: " + code);
		}
		return n.name;
	}
}
