# Required functions to run simulations

"%+%" <- function(x,y){paste(x,y,sep="")}

complete_gsub <- function(command,paramlist) {
  for (p in names(paramlist)) command = gsub('[' %+% p %+% ']',paramlist[[p]],command,fixed=T)
  command
}
