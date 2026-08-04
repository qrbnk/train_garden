const TRAIN_GARDEN_DB = {
    // Mapping diagnostic answers to roles
    roleMappings: [
        { strength: "Strategy & Planning", environment: "Corporate / Office", role: "Sustainability Consultant", salary: "£45,000 - £65,000", skills: ["Carbon Accounting", "Project Management", "ESG Reporting"] },
        { strength: "Technical Troubleshooting", environment: "Outdoor / Field", role: "Solar PV Engineer", salary: "£35,000 - £50,000", skills: ["Electrical Engineering", "Blueprint Reading", "Safety Compliance"] },
        { strength: "People & Communication", environment: "Hybrid / Flexible", role: "Sustainability Engagement Officer", salary: "£30,000 - £42,000", skills: ["Public Speaking", "Community Outreach", "Content Creation"] },
        { strength: "Physical / Craftsmanship", environment: "Outdoor / Field", role: "Retrofit Technician", salary: "£28,000 - £38,000", skills: ["Insulation Installation", "Thermal Imaging", "Carpentry"] }
    ],

    // Skills database for comparison
    skillsLibrary: [
        "Project Management", "Data Analysis", "Public Speaking", "Electrical Engineering", 
        "Customer Service", "Report Writing", "Strategic Planning", "Safety Compliance"
    ],

    // Volunteering Opportunities
    volunteering: [
        {
            id: 1,
            title: "Community Garden Coordinator",
            email: "volunteers@urbangreens.org",
            skillsGained: ["Community Outreach", "Project Management"],
            description: "Help manage a local green space and coordinate volunteers."
        },
        {
            id: 2,
            title: "Retrofit Apprentice Assistant",
            email: "careers@retrofituk.co.uk",
            skillsGained: ["Insulation Installation", "Safety Compliance"],
            description: "Work alongside seniors to learn building decarbonization."
        }
    ]
};