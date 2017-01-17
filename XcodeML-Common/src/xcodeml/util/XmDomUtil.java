/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import org.w3c.dom.*;
import java.util.ArrayList;
import java.util.HashSet;

public class XmDomUtil {
    public static Node getElement(Node n, String name) {
        if (n == null)
            return null;

        NodeList list = n.getChildNodes();
        for (int i = 0; i < list.getLength(); i++) {
            Node thisNode = list.item(i);
            if (thisNode.getNodeType() != Node.ELEMENT_NODE)
                continue;
            if (thisNode.getNodeName().equals(name))
                return thisNode;
        }
        return null;
    }

    public static Node getContent(Node n) {
        if (n == null)
            return null;

        NodeList list = n.getChildNodes();
        for (int i = 0; i < list.getLength(); i++) {
            Node thisNode = list.item(i);
            if (thisNode.getNodeType() != Node.ELEMENT_NODE)
                continue;
            return thisNode;
        }
        return null;
    }

    public static String getContentText(Node n) {
        if (n == null)
            return null;
        if (n.getFirstChild() == null) {
            return "";
        }

        return n.getFirstChild().getNodeValue();
    }

    public static String getAttr(Node n, String name) {
        if (n == null)
            return null;

        Node nn = n.getAttributes().getNamedItem(name);
        if (nn != null)
            return nn.getNodeValue();
        return null;
    }

    public static boolean getAttrBool(Node n, String name) {
        if (n == null)
            return false;

        Node nn = n.getAttributes().getNamedItem(name);
        if (nn == null)
            return false;

        String value = nn.getNodeValue();

        if (value == null)
            return false;

        if ("true".equals(value.toLowerCase()))
            return true;

        if ("1".equals(value.toLowerCase()))
            return true;

        return false;
    }

    public static Node[] collectElementsAsArray(Node n, String name) {
        if (n == null) {
            return new Node[0];
        }

        return collectElements(n, name).toArray(new Node[0]);
    }

    public static ArrayList<Node> collectElements(Node n, String ... names) {
        if (n == null) {
            return null;
        }

        NodeList list = n.getChildNodes();
        ArrayList<Node> nodes = new ArrayList<Node>();
        for (int i = 0; i < list.getLength(); i++) {
            Node childNode = list.item(i);
            if (childNode.getNodeType() != Node.ELEMENT_NODE)
                continue;
            for (String name : names) {
                if (childNode.getNodeName().equals(name)) {
                    nodes.add(childNode);
                }
            }
        }
        return nodes;
    }

    public static ArrayList<Node> collectElementsExclude(Node n,
                                                         String ... names) {
        if (n == null) {
            return null;
        }

        HashSet<String> excludes = new HashSet<String>();
        for (String name : names) {
            excludes.add(name);
        }

        NodeList list = n.getChildNodes();
        ArrayList<Node> nodes = new ArrayList<Node>();
        for (int i = 0; i < list.getLength(); i++) {
            Node childNode = list.item(i);
            if (childNode.getNodeType() != Node.ELEMENT_NODE)
                continue;
            if (!excludes.contains(childNode.getNodeName())) {
                nodes.add(childNode);
            }
        }
        return nodes;
    }

    public static ArrayList<Node> collectChildNodes(Node n) {
        if (n == null) {
            return null;
        }

        NodeList list = n.getChildNodes();
        ArrayList<Node> nodes = new ArrayList<Node>();
        for (int i = 0; i < list.getLength(); i++) {
            Node childNode = list.item(i);
            if (childNode.getNodeType() != Node.ELEMENT_NODE)
                continue;
            nodes.add(childNode);
        }
        return nodes;
    }
}
