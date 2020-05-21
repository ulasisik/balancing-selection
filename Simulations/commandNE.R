args <- commandArgs(T)
dirs = file.path(as.character(args[1]), "Params.txt")
funs = file.path(as.character(args[1]), "simFunc.R")

source(dirs)
source(funs)

s = as.numeric(args[2])  
h = as.numeric(args[3])
r = as.numeric(args[4])
start=as.numeric(args[5])

for(i in start:r){
  pr = list(
    mu = mu,
    recrate = rnorm(1, m=1e-8, sd=1e-8/10),  #recombination rate(r)
    L = L_NE,   
    N_A = N_A,
    N_AF = N_AF,
    N_B = N_B,
    N_EU0 = N_EU0,
    N_AS0 = N_AS0,
    N_end = N_end,
    T_AF = round(start_time/gene_time - 312000/gene_time),
    T_B = round(start_time/gene_time - 125000/gene_time),
    T_EUAS = round(start_time/gene_time - 42300/gene_time),
    Tend = round(start_time/gene_time),  #generation number at which simulation ends(today)
    m_AFB = m_AFB,
    m_AFEU = m_AFEU,
    m_EUAS = m_EUAS,
    m_AFAS = m_AFAS,
    grate_eu = grate_eu,   #growth rates
    grate_as = grate_as,
    datadir = DIRDATA,
    h = h,
    s = s,
    r=i)
  
  
  sim=paste('initialize() {',  
            'initializeMutationRate([mu]); ',
            'initializeMutationType("m1", 0.5, "f", 0.0); ',
            'm1.convertToSubstitution = F;',
            'initializeGenomicElementType("g1", m1, 1.0); ',
            'initializeGenomicElement(g1, 0, [L]); ',
            'initializeRecombinationRate([recrate]);',
            '}',
            '1 { sim.addSubpop("p1", [N_A]); }',
            '[T_AF] { p1.setSubpopulationSize([N_AF]); }',
            '[T_B] { sim.addSubpopSplit("p2", [N_B], p1); ',
            'p1.setMigrationRates(c(p2), c([m_AFB])); ',
            'p2.setMigrationRates(c(p1), c([m_AFB]));',
            '}',
            '[T_EUAS] { ',
            'sim.addSubpopSplit("p3", [N_AS0], p2); ',
            'p2.setSubpopulationSize([N_EU0]);',
            
            'p1.setMigrationRates(c(p2, p3), c([m_AFEU], [m_AFAS])); ',
            'p2.setMigrationRates(c(p1, p3), c([m_AFEU], [m_EUAS])); ',
            'p3.setMigrationRates(c(p1, p2), c([m_AFAS], [m_EUAS]));',
            '}',
            '[T_EUAS]:[Tend] {',
            't = sim.generation - [T_EUAS]; ',
            'p2_size = round([N_EU0] * exp([grate_eu] * t)); ',
            'p3_size = round([N_AS0]* exp([grate_as] * t));',
            
            'p2.setSubpopulationSize(asInteger(p2_size)); ',
            'p3.setSubpopulationSize(asInteger(p3_size));',
            '}',
            '[Tend] late() { ',
            'p2.outputMSSample([N_end], replace = F, filePath="[datadir]NE_[r].txt");',
            '}', 
            sep = "\n")
  
  f= file(paste(paste0(DIRTMP,"NE"),i, sep="_"),open = "w")
  sim=complete_gsub(sim,pr)
  write(sim, file=f, append = T)
  close(f)
}
