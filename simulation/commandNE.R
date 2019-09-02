args <- commandArgs(T)

source("/mnt/NEOGENE1/projects/deepLearn_selection_2018/balancing-selection/simulation/simFunc.R")

s = as.numeric(args[1])  #sample 100 diploid individual
h = as.numeric(args[2])
r = as.numeric(args[3])
start=as.numeric(args[4])

gene_time = 29 #generation time (Tremblay and Vézina 2000)
start_time = 602000 #start 1000 generation (~25 kya) before first event(which is increase N)

for(i in start:r){
  pr = list(
    mu = 1.44e-8,  #mutation rate
    recrate = rnorm(1, m=1e-8, sd=1e-8/10),  #recombination rate(r)
    L = 500000,    #length(bp)
    N_A = 11273,
    N_AF = 23721,
    N_B = 3104,
    N_EU0 = 2271,
    N_AS0 = 924,
    N_end = 198,
    T_AF = round(start_time/gene_time - 312000/gene_time),
    T_B = round(start_time/gene_time - 125000/gene_time),
    T_EUAS = round(start_time/gene_time - 42300/gene_time),
    Tend = round(start_time/gene_time),  #generation number at which simulation ends(today)
    m_AFB = 15.8e-5,
    m_AFEU = 1.1e-5,
    m_EUAS = 4.19e-5,
    m_AFAS = 0.48e-5,
    grate_eu = 0.00196,   #growth rates
    grate_as = 0.00309,
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
            'p2.outputMSSample([N_end], replace = F, filePath="/mnt/NEOGENE1/projects/deepLearn_selection_2018/50kb/results/decompMSMS/NE_[r].txt");',
            '}', 
            sep = "\n")
  
  f= file(paste("/mnt/NEOGENE1/projects/deepLearn_selection_2018/50kb/scripts/simulations/SLiM_scripts/NE",i, sep="_"),open = "w")
  sim=complete_gsub(sim,pr)
  write(sim, file=f, append = T)
  close(f)
}
