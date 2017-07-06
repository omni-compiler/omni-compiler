package xcodeml.c.type;

/**
 * interface for sturct/union/enum type.
 */
public interface XcTaggedType
{
    public String getTagName();

    public String getTypeNameHeader();

    public void setTagName(String tagName);
}
