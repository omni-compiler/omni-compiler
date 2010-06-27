/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.util.ArrayList;

import xcodeml.XmObj;

/**
 * The matcher execute OR matchings.
 */
public class XmObjOrCompositeMatcher implements XmObjMatcher
{
    private ArrayList<XmObjMatcher> list;

    /**
     * Creates XmObjORCompositeMatcher
     */
    public XmObjOrCompositeMatcher()
    {
        list = new ArrayList<XmObjMatcher>();
    }

    @Override
    public boolean match(XmObj n)
    {
        for(XmObjMatcher m : list) {
            if(m.match(n)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Adds matcher be executed by this matcher.
     *
     * @param matcher provides condition.
     */
    public void addMatcher(XmObjMatcher matcher)
    {
        list.add(matcher);
    }
}
