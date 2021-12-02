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

            // TODO: target, target teams, target parallel, ...
            switch (p) {
            case TARGET:
            case TARGET_DATA:
            case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP:
                return true;
            }
        }

        return false;
    }

    public boolean isInTaskOffloadWithForLoop() {
        Iterator<OMPpragma> it = stack.iterator();

        // 'stack' contains self.
        // So, exclude self from the search.
        it.next();

        while (it.hasNext()) {
            OMPpragma p = (OMPpragma) it.next();

            // TODO: distribute parallel for, distribute,
            //       for, target teams distribute, target parallel for,
            //       teams distributea, teams distribute parallel for
            switch (p) {
            case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP:
            case PARALLEL_FOR:
                return true;
            }
        }

        return false;
    }
}
