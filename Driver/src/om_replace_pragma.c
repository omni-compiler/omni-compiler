#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define PRAGMA_NAME0 "_OM_PRAGMA_"
#define PRAGMA_NAME1 "_OM_PRAGMA"
#define IS_VALID_STR(s)  ((s) != NULL && *(s) != '\0')

#define SYSTEM_INCLUDE 1
#define USER_INCLUDE 2
#define BUFFER_SIZE 1

typedef struct{
  FILE *file;
  int curLine;
  char *filename;
  char *dir;
  int numCommentedLFs;
  int numEscapedLFs;
  fpos_t pos;
  char inStrLiteral;
} SrcFile;

static FILE *output;
static char enableXMP = 1;
static char staticBuffer[BUFFER_SIZE];
static char **includeDirs;
static int numIncludeDirs;
static int maxIncludeDirs;
static char **srcFileNameStack;
static int srcFileNameStackSize;
static int maxSrcFileNameStackSize;

static void ungetChar_(int c, SrcFile *sf){ 
  if(c == '\n') (sf->curLine)--;
  ungetc(c, sf->file);
}

//return char except disabled-return
static int getChar_(SrcFile *sf)
{
  int c;
  while( (c = getc(sf->file)) == '\\' ){
    int nc = getc(sf->file);
    if(nc == '\n'){
      sf->numEscapedLFs++;
      (sf->curLine)++;
    }else{
      ungetChar_(nc, sf);
      break;
    }
  }

  if(c == '\n'){
    (sf->curLine)++;
    while(sf->numEscapedLFs > 0){
      ungetChar_('\n', sf);
      sf->numEscapedLFs--;
    }
  }

  return c;
}

static int skipBlockComment(SrcFile *sf){
  int startLine = sf->curLine;
  int numCommentedLFs = 0;
  int c = getChar_(sf);

  while( c != EOF){
    int nc = getChar_(sf);
    if(c == '*' && nc == '/'){
      return numCommentedLFs;
    }
    if(c == '\n') numCommentedLFs++;
    c = nc;
  }

  //reached EOF
  fprintf(stderr, "error: %s: line %d, unterminated comment\n", sf->filename, startLine);
  exit(EXIT_FAILURE);
}

static void skipSingleLineComment(SrcFile *sf)
{
  int c;
  while( (c = getChar_(sf)) != EOF){
    if(c == '\n') break;
  }
}

//return char except comments
static int getChar(SrcFile *sf)
{
  int c = getChar_(sf);

  if(c == '\\'){
    int nc = getChar_(sf);
    if(nc == '"'){ //escaped double-quotation
      sf->inStrLiteral = !sf->inStrLiteral;
    }
    ungetChar_(nc, sf);
  }else if(sf->inStrLiteral){
    if(c == '"'){
      sf->inStrLiteral = 0;
    }
  }else{
    if(c == '/'){
      int nc = getChar_(sf);
      if(nc == '*'){ // block comment
	sf->numCommentedLFs += skipBlockComment(sf);
	c = ' ';
      }else if(nc == '/'){ // single-line comment
	skipSingleLineComment(sf);
	c = '\n';
      }else{
	ungetChar_(nc, sf);
      }
    }else if(c == '"'){
      sf->inStrLiteral = 1;
    }
  }

  if(c == '\n'){
    while(sf->numCommentedLFs > 0){
      ungetChar_('\n', sf);
      sf->numCommentedLFs--;
    }
  }
  return c;
}

static char* getLine(SrcFile *sf){
  int c = getChar(sf);
  char *buf = staticBuffer;
  char *p = buf;

  if(c == EOF){
    return NULL;
  }
  
  int numChars = 0;
  while( c != EOF && c != '\n' && numChars < BUFFER_SIZE - 1){
    *p++ = (char)c;
    c = getChar(sf);
    ++numChars;
  }

  int bufferSize = BUFFER_SIZE;
  while(numChars == bufferSize - 1){
    bufferSize += BUFFER_SIZE;
    char *newBuf;
    if(buf == staticBuffer){
      newBuf = (char*)malloc(bufferSize * sizeof(char));
    }else{
      newBuf = (char*)realloc(buf, bufferSize * sizeof(char));
    }
    if(newBuf == NULL){
      fprintf(stderr, "error: cannot realloc buffer\n");
      exit(EXIT_FAILURE);
    }
    if(buf == staticBuffer){
      strncpy(newBuf, staticBuffer, numChars);
    }
    buf = newBuf;
    p = newBuf + numChars;
    while( c != EOF && c != '\n' && numChars < bufferSize - 1){
      *p++ = (char)c; 
      c = getChar(sf);
      ++numChars;
    }      
  }

  *p = '\0';
  return buf;
}

static char isBlank(char c){
  return (c == ' ' || c == '\t');
}

static char *skipBlank(char *p)
{
  while( isBlank(*p) ) ++p;
  return p;
}

static char *skipWord(char *p)
{
  while( !isBlank(*p) ) ++p;
  return p;
}

static int startsWithWord(char *p, char *w)
{
  int wordLen = strlen(w);
  if( strncmp(p, w, wordLen) != 0 ) return 0;
  if( ! isBlank(p[wordLen]) ) return 0;
  return 1;
}

static char isSharp(char c){
  return c == '#';
}

static char *skipSharp(char *p)
{
  while( *p == '#') ++p;
  return p;
}

static char isPragma(char *p)
{
  p = skipBlank(p);
  if( !isSharp(*p++) ) return 0;
  p = skipBlank(p);
  if( !startsWithWord(p, "pragma") ) return 0;
  return 1;
}

static char *getPragmaStr(char *p)
{
  return skipBlank(skipWord(skipBlank(skipSharp(skipBlank(p)))));
}


static int isInclude(char *p)
{
  p = skipBlank(p);
  if( !isSharp(*p++) ) return 0;
  p = skipBlank(p);
  if( strncmp(p, "include", 7) != 0 ) return 0;
  p += 7;
  p = skipBlank(p);
  
  if( *p != '"' &&  *p != '<') return 0;
  int isSystemInclude = (*p == '<');
  char terminator = (isSystemInclude)? '>' : '"';
  char *lastTerminatorP = NULL; //last terminator's pointer
  //find last terminator
  while( *++p != '\0' ){
    if( *p == terminator ) lastTerminatorP = p;
  }
  if(lastTerminatorP == NULL) return 0;
  p = lastTerminatorP;
  while(*++p != '\0'){
    if( !isBlank(*p) ) return 0;
  }
  return (isSystemInclude)? SYSTEM_INCLUDE : USER_INCLUDE;
}

static char *getIncludeFileName(char *p)
{
  char *filename = (char*)malloc(sizeof(char)*(strlen(p)+1));
  if(filename == NULL){
    fprintf(stderr, "error: cannot alloc memory for includeFileName\n");
    exit(EXIT_FAILURE);
  }
  
  //if( (p = strpbrk(p, "<\"")) == NULL) return NULL;
  while( *p != '<' && *p != '"') ++p;
  ++p;
  char *q = filename;
  while( (*q++ = *p++) );
  --q;
  while( *q != '>' && *q != '"') --q;
  *q = '\0';
  return filename;
}

static int exists(char *filePath)
{
  struct stat st;
  return (stat(filePath, &st) == 0);
}

static char *findIncludeFilePath(char *srcDir, char *includeFileName, int includeKind)
{
  char *filePath = (char*)malloc(sizeof(char) * FILENAME_MAX);
  if(filePath == NULL){
    fprintf(stderr, "error: cannot alloc memory for filePath\n");
    exit(EXIT_FAILURE);
  }
  char **incDirs = includeDirs;

  if(includeKind == USER_INCLUDE){
    strcpy(filePath, srcDir);
    strcat(filePath, includeFileName);
    if(exists(filePath)){
      return filePath;
    }
  }
  
  while(*incDirs != NULL){
    char *incDir = *incDirs;
    char *p = filePath;
    while( (*p++ = *incDir++) );
    --p;
    if( *(p-1) != '/'){
      *p = '/';
      ++p;
    }

    char *f = includeFileName;
    while( (*p++ = *f++) );
    if(exists(filePath)){
      return filePath;
    }
    incDirs++;
  }
  free(filePath);
  return NULL;
}

static char *getDir(char *filePath){
  int i = 0;
  char *dir = (char*)malloc(sizeof(char)*(strlen(filePath)+1));
  if(dir == NULL){
    fprintf(stderr, "error: cannot alloc memory for dir\n");
  }
  while( (dir[i] = filePath[i]) ) i++;
  --i;
  for(; i >= 0 && dir[i] != '/'; i--);
  dir[++i] = '\0';
  return dir;
}

static void pushSrcFileName(char *srcFileName)
{
  const int increment = 16;
  //printf("push:\"%s\"\n", srcFileName);
  if(srcFileNameStackSize == maxSrcFileNameStackSize){
    maxSrcFileNameStackSize += increment;
    char **newStack = (char **)realloc(srcFileNameStack, (maxSrcFileNameStackSize + 1) * sizeof(char*));
    if(newStack == NULL){
      fprintf(stderr, "error: cannot realloc srcFileNameStack\n");
      exit(EXIT_FAILURE);
    }
    srcFileNameStack = newStack;
  }
  srcFileNameStack[srcFileNameStackSize++] = srcFileName;
  srcFileNameStack[srcFileNameStackSize] = NULL;
}

static void popSrcFileName()
{
  //printf("pop:\"%s\"\n", srcFileNameStack[srcFileNameStackSize -1]);
  srcFileNameStack[--srcFileNameStackSize] = NULL;
}

static int findProcessedFileName(char *srcFileName)
{
  char **srcFileNames = srcFileNameStack;
  while(*srcFileNames != NULL){
    if(strcmp(*srcFileNames, srcFileName) == 0){
      return 1;
    }
    srcFileNames++;
  }
  return 0;
}

static void preprocess(char *srcFileName){
  char *buf;
  //fprintf(stderr, "pp \"%s\"\n", srcFileName);
  //init srcFile
  if(findProcessedFileName(srcFileName)){
    fprintf(stderr, "error: \"%s\" was included more than once\n", srcFileName);
    exit(EXIT_FAILURE);
  }
  SrcFile *src = malloc(sizeof(SrcFile));
  if(src == NULL){
    fprintf(stderr, "error: cannot alloc structure\n");
  }
  src->filename = srcFileName;
  src->dir = getDir(srcFileName);
  src->curLine = 1;
  src->numCommentedLFs = 0;
  src->numEscapedLFs = 0;
  src->inStrLiteral = 0;
  src->file = fopen(src->filename, "r");
  if(src->file == NULL){
    fprintf(stderr, "error: cannot open \"%s\"\n", src->filename);
    exit(EXIT_FAILURE);
  }
  pushSrcFileName(srcFileName);

  while( (buf = getLine(src)) != NULL){
    int includeKind;
    if(isPragma(buf)){
      char *pragmaStr = getPragmaStr(buf);

      if( startsWithWord(pragmaStr, "omp") || startsWithWord(pragmaStr, "acc")
	  || (enableXMP && startsWithWord(pragmaStr, "xmp")) ){
	fprintf(output, "%s(%s)\n", PRAGMA_NAME1, pragmaStr);
      }else{
	fprintf(output, "%s\n", buf);
      }
    }else if( (includeKind = isInclude(buf)) ){
      char *includeFileName = getIncludeFileName(buf);
      char *includeFilePath = findIncludeFilePath(src->dir, includeFileName, includeKind);
      if(includeFilePath != NULL){
	//get pos
	fgetpos(src->file, &(src->pos));
	//close current file
	fclose(src->file);

	//output filename
	fprintf(output, "# 1 \"%s\"\n", includeFilePath);
	preprocess(includeFilePath);
	fprintf(output, "# %d \"%s\"\n", src->curLine , src->filename);

	//reopen
	src->file = fopen(src->filename, "r");
	if(src->file == NULL){
	  fprintf(stderr, "error: cannot reopen \"%s\"\n", src->filename);
	  exit(EXIT_FAILURE);
	}

	//set pos
	fsetpos(src->file, &(src->pos));
	free(includeFilePath);
      }else{
	if(includeKind == USER_INCLUDE){
	  fprintf(output, "#include \"%s\"\n", includeFileName);
	}else{
	  fprintf(output, "#include <%s>\n", includeFileName);
	}
      }
      free(includeFileName);
    }else{
      fprintf(output, "%s\n", buf);
    }
    if(buf != staticBuffer){
      free(buf);
    }
  }
  
  popSrcFileName();

  fclose(src->file);
  free(src->dir);
  free(src);
}


static void addIncludeDir(char *dir)
{
  const int increment = 16;
  if(numIncludeDirs == maxIncludeDirs){
    maxIncludeDirs += increment;
    char **newDirs = (char**)realloc(includeDirs, (maxIncludeDirs + 1) * sizeof(char*));
    if(newDirs == NULL){
      fprintf(stderr, "error: cannot realloc includeDirs\n");
      exit(EXIT_FAILURE);
    }
    includeDirs = newDirs;
  }
  includeDirs[numIncludeDirs++] = dir;
  includeDirs[numIncludeDirs] = NULL;
}

int main(int argc, char* argv[])
{
  output = stdout;
  const int initialMaxIncludeDirs = 16;
  const int initialMaxSrcFileNameStackSize = 16;
  char *input_filename = NULL;
  char *output_filename = NULL;

  includeDirs = (char**)malloc( (initialMaxIncludeDirs + 1) * sizeof(char*));
  if(includeDirs ==NULL){
    fprintf(stderr, "error: cannot alloc includeDirs\n");
    exit(EXIT_FAILURE);
  }
  numIncludeDirs = 0;
  maxIncludeDirs = initialMaxIncludeDirs;
  includeDirs[0] = NULL;

  srcFileNameStack = (char**)malloc( (initialMaxSrcFileNameStackSize + 1) * sizeof(char*));
  if(includeDirs ==NULL){
    fprintf(stderr, "error: cannot alloc srcFileNameStack\n");
    exit(EXIT_FAILURE);
  }
  srcFileNameStackSize = 0;
  maxSrcFileNameStackSize = initialMaxSrcFileNameStackSize;
  srcFileNameStack[0] = NULL;

  argv++;
  while(*argv != NULL){
    if(strcmp(*argv, "--no-replace-xmp") == 0){
      enableXMP = 0;
    }else if(strcmp(*argv, "-o") == 0){
      if(output_filename != NULL){
	fprintf(stderr, "error: too many output\n");
	exit(EXIT_FAILURE);
      }
      if(IS_VALID_STR(*(argv+1))){
	output_filename = *(argv+1);
	argv++;
      }else{
	fprintf(stderr, "error: invalid output\n");
      }
    }else if(strncmp(*argv, "-I", 2) == 0){
      addIncludeDir(*argv + 2);
    }else{
      if(input_filename != NULL){
	fprintf(stderr, "error: too many input\n");
	exit(EXIT_FAILURE);
      }
      if(IS_VALID_STR(*argv)){
	input_filename = *argv;
      }else{
	fprintf(stderr, "error: invalid input\n");
      }
    }
    argv++;
  }

  if(input_filename == NULL){
    input_filename = "<stdin>";
  }

  if(output_filename != NULL){
    output = fopen(output_filename, "w");
    if(output == NULL){
      fprintf(stderr, "error: cannot open %s\n", output_filename);
      exit(EXIT_FAILURE);
    }
  }else{
    output_filename = "<stdout>";
  }

  
  //output OM_PRAGMA def
  fprintf(output, "#define " PRAGMA_NAME0 "(...) _Pragma(#__VA_ARGS__)\n");
  fprintf(output, "#define " PRAGMA_NAME1 "(...) " PRAGMA_NAME0 "(__VA_ARGS__)\n");

  //preprocess
  fprintf(output, "# 1 \"%s\"\n", input_filename);
  preprocess(input_filename);

  free(includeDirs);
  free(srcFileNameStack);
  
  fflush(output);
  if(output != stdout){
    fclose(output);
  }

  return 0;
}
