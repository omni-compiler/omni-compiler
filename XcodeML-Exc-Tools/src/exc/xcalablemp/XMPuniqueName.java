package exc.xcalablemp;

public class XMPuniqueName {
  private final static int BUFFER_LEN = 32;
  private static XMPuniqueName _instance = null;

  private char[] _buffer;
  private int _pointer;

  private XMPuniqueName() {
    _buffer = new char[BUFFER_LEN];

    _pointer = 0;
    _buffer[0] = 'a';
  }

  private void changeStatus() throws XMPexception {
    changeStatus(0);
  }

  private void changeStatus(int p) throws XMPexception {
    if (p > _pointer) {
      _pointer++;
      if (_pointer == BUFFER_LEN)
        throw new XMPexception("cannot create a unique name");
      else {
        _buffer[_pointer] = 'a';
        return;
      }
    }

    if ((_buffer[p] >= 'a') && (_buffer[p] <= 'y')) {
      _buffer[p]++;
    }
    else if ((_buffer[p] >= 'A') && (_buffer[p] <= 'Y')) {
      _buffer[p]++;
    }
    else if ((_buffer[p] >= '0') && (_buffer[p] <= '8')) {
      _buffer[p]++;
    }
    else {
      switch (_buffer[p]) {
        case 'z':
          { _buffer[p] = 'A'; }
          break;
        case 'Z':
          { _buffer[p] = '0'; }
          break;
        case '9':
          {
            _buffer[p] = 'a';
            changeStatus(p + 1);
          }
          break;
        default:
          throw new XMPexception("cannot create a unique name");
      }
    }
  }

  public static String getUniqueName() throws XMPexception {
    if (_instance == null) _instance = new XMPuniqueName();
    else                   _instance.changeStatus();

    return new String(_instance._buffer, 0, _instance._pointer + 1);
  }
}
