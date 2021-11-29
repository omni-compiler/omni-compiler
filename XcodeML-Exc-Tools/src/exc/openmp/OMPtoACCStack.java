package exc.openmp;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;

public class OMPtoACCStack {

    private Deque<OMPpragma> stack = new ArrayDeque<>();

    public OMPtoACCStack() {
    }

    public void push(OMPpragma p) {
        stack.push(p);
    }

    public OMPpragma pop() {
        return stack.pop();
    }

    public boolean isInTaskOffload() {
        Iterator<OMPpragma> it = stack.iterator();

        // 'stack' contains self.
        // So, exclude self from the search.
        it.next();

        while (it.hasNext()) {
            OMPpragma p = (OMPpragma) it.next();

            // TODO: target, target teams, target parallel
            switch (p) {
            case TARGET_DATA:
                return true;
            }
        }

        return false;
    }
}
