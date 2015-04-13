package exc.object;

import java.util.Vector;

/**
 * Super class of objects which has a property list. The property list is
 * implmented by a Vector.
 */
public class PropObject
{
    Vector<Object> props; // property list
    protected int id; // identifier

    public int Id()
    {
        return id;
    }

    public void setId(int id)
    {
        this.id = id;
    }

    public void setProp(Object key, Object value)
    {
        if(props == null)
            props = new Vector<Object>();
        int index = props.indexOf(key);
        if(index == -1) {
            props.addElement(key);
            props.addElement(value);
        } else {
            props.setElementAt(value, index + 1);
        }
    }

    public Object getProp(Object key)
    {
        if(props == null)
            return null;
        int index = props.indexOf(key);
        if(index == -1)
            return null;
        else
            return props.elementAt(index + 1);
    }

    public void remProp(Object key)
    {
        if(props == null)
            return;
        int index = props.indexOf(key);
        if(index == -1)
            return;
        else {
            props.removeElementAt(index);
            props.removeElementAt(index);
        }
    }

    public void remProperties()
    {
        props = null;
    }

    public Vector<Object> getProperties()
    {
        return props;
    }

    public void setProperties(Vector<Object> props)
    {
        this.props = props;
    }
}
