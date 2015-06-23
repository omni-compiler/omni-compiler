#!/bin/sh

## Set Arguments
OUTPUT_DIR=$1
JOB_NUM=$2
ELAPSED_TIME_SEC=$3
BASE_TESTDIR=$4
SORTED_LIST=$5
shift; shift; shift; shift; shift
TESTDIRS=$@
CURRENT_DIR=`pwd`

## Set static variable
TIMELINE_FILE=timeline.dat
STATIC_COLORS=('red' 'darkred' 'mistyrose' 'steelblue' 'lightsteelblue' 'darkblue' 'yellowgreen' 'darkolivegreen' 'lime' 'magenta' 'darkmagenta' 'mediumpurple' '#cbb69d' '#603913' '#c69c6e')
NUM_OF_STATIC_COLORS=${#STATIC_COLORS[*]}
TIME_LINE_DATA=""
NUM_OF_DIRS=0
TIMELINE_WIDTH=1500

## Set Data of master
for dat in autogen.dat configure.dat make.dat; do
    FILE=${CURRENT_DIR}/$dat
    if [ -f ${FILE} ]; then
	JOB_NAME=`head -1 ${FILE}`
	NODE_NAME=`head -2 ${FILE} | tail -1`
	START_TIME=`head -3 ${FILE} | tail -1`
	END_TIME=`tail -1 ${FILE}`
	TIME_LINE_DATA+="['"${NODE_NAME}"',\t"
	TIME_LINE_DATA+="'"${JOB_NAME}"',\t"
	TIME_LINE_DATA+=" new Date(${START_TIME}),\t"
	TIME_LINE_DATA+=" new Date(${END_TIME})],\n"
	NUM_OF_DIRS=`expr $NUM_OF_DIRS + 1`
    fi
done

## Set Data of slave
NODE_NAME_LIST=""
for subdir in ${TESTDIRS}; do
    FILE=${BASE_TESTDIR}/$subdir/${TIMELINE_FILE}
    if [ -f ${FILE} ]; then
	JOB_NAME=`head -1 ${FILE}`
	NODE_NAME=`head -2 ${FILE} | tail -1`
	NODE_NAME_LIST+="$NODE_NAME\n"
	START_TIME=`head -3 ${FILE} | tail -1`
	END_TIME=`tail -1 ${FILE}`
	TIME_LINE_DATA+="['"${NODE_NAME}"',\t"
	TIME_LINE_DATA+="'"${JOB_NAME}"',\t"
	TIME_LINE_DATA+=" new Date(${START_TIME}),\t"
	TIME_LINE_DATA+=" new Date(${END_TIME})],\n"
	NUM_OF_DIRS=`expr $NUM_OF_DIRS + 1`
    fi
done
TIME_LINE_DATA=`echo -e $TIME_LINE_DATA | sort -n`

## Calculate height of timeline graph
# Note that NODE_NAME_LIST has "\n" in the last line
# So that when number of Slaves is 4, the value of NODE_NAME_LIST is 5.
# Additiolal 1 is regard as Master.
# The height is "45px * (num of slaves + master) + 30."
NUM_OF_NODES=`echo -e $NODE_NAME_LIST | sort | uniq | wc -l`
TIMELINE_HEIGHT=`expr 45 \* $NUM_OF_NODES`
TIMELINE_HEIGHT=`expr 30 + $TIMELINE_HEIGHT`

## Create Directory
mkdir -p $OUTPUT_DIR
OUTPUT_FILE=${OUTPUT_DIR}"/"${JOB_NUM}.html

## Set bar colors
COLORS=""
NUM=0
for subdir in `seq 1 $NUM_OF_DIRS`; do
    REST_NUM=`expr $NUM % $NUM_OF_STATIC_COLORS`
    COLORS+="'"${STATIC_COLORS[${REST_NUM}]}"',"
    NUM=`expr $NUM + 1`
done

## Set Sorted list
MAXLINE=`wc -l ${SORTED_LIST} | awk '{print $1}'`
CONTENTS=""

for line in $(seq 1 $MAXLINE); do
    priority=`head -n $line ${SORTED_LIST} | tail -1 | awk '{print $1}'`
    dirname=`head -n $line ${SORTED_LIST} | tail -1 | awk '{print $2}'`
    CONTENTS="${CONTENTS}  <tr><td align=\"right\">${priority}</td><td>${dirname}</td></tr>\n"
done

SORTED_LIST_HTML=`cat <<EOF
<table class="sample">\n
  <tr><th align="right">Num of files (Priority)</th><th>Directory Name</th></tr>\n
   ${CONTENTS}
</table>\n
EOF`

## OUTPUT
OUTPUT=`cat <<EOF
<html>\n
<head>\n
<link rel="stylesheet" type="text/css" href="style.css"\n
</head>\n
<body>\n
<script type="text/javascript" src="https://www.google.com/jsapi?autoload={'modules':[{'name':'visualization', \n
                                    'version':'1','packages':['timeline']}]}"></script> \n
<script type="text/javascript"> \n
\n
google.setOnLoadCallback(drawChart);\n
function drawChart() \n
{\n
    var container = document.getElementById('timeline');\n
    var chart = new google.visualization.Timeline(container);\n
\n
    var dataTable = new google.visualization.DataTable();\n
    dataTable.addColumn({ type: 'string', id: 'Position' });\n
    dataTable.addColumn({ type: 'string', id: 'JobName' });\n
    dataTable.addColumn({ type: 'date', id: 'Start' });\n
    dataTable.addColumn({ type: 'date', id: 'End' });\n
    dataTable.addRows([\n
    ${TIME_LINE_DATA}
    ]);\n
\n
    var options = {\n
    hAxis:{ format:'kk:mm:ss'},\n
    colors: [\n
    ${COLORS}\n
    ]};\n
\n
    chart.draw(dataTable, options);\n
  }\n
</script>\n
\n
<div id="timeline" style="width: ${TIMELINE_WIDTH}px; height: ${TIMELINE_HEIGHT}px;"></div>\n
<p>\n
Elapse time is $ELAPSED_TIME_SEC sec.\n
</p>\n
<p>\n
${SORTED_LIST_HTML}
</p>\n
</body>\n
</html>\n
EOF`

echo -e ${OUTPUT} > ${OUTPUT_FILE}
